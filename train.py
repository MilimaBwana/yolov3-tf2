from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks,
    convert_yolo_output
)
from yolov3_tf2.utils import freeze_all, create_detections, create_annotations, clear_directory
from yolov3_tf2.metrics import average_precisions, calculate_map
import yolov3_tf2.dataset as dataset
from yolov3_tf2.checkpoint_handler import BestEpochCheckpoint

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_boolean('augmentation', False, 'Enable image augmentation during training')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                                                  'useful in transfer learning with different number of classes')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    train_dataset = dataset.load_fake_dataset()
    if FLAGS.dataset:
        train_dataset = dataset.load_tf_record(tfrecord=FLAGS.dataset, mode=tf.estimator.ModeKeys.TRAIN,
                                               class_file=FLAGS.classes, anchors=anchors,
                                               anchor_masks=anchor_masks, batch_size=FLAGS.batch_size,
                                               max_detections=FLAGS.yolo_max_boxes, size=FLAGS.size,
                                               augmentation=FLAGS.augmentation)

    val_dataset = dataset.load_fake_dataset()
    if FLAGS.val_dataset:
        val_dataset = dataset.load_tf_record(tfrecord=FLAGS.val_dataset, mode=tf.estimator.ModeKeys.EVAL,
                                             class_file=FLAGS.classes, anchors=anchors,
                                             anchor_masks=anchor_masks, batch_size=FLAGS.batch_size,
                                             max_detections=FLAGS.yolo_max_boxes, size=FLAGS.size,
                                             augmentation=False)

    # Configure the model for transfer learning
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else:
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))

        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)

    else:
        # All other transfer require matching classes
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    train_log_dir = './logs/train'
    val_log_dir = './logs/valid'
    clear_directory(train_log_dir, clear_subdirectories=True)
    clear_directory(val_log_dir, clear_subdirectories=True)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # Define checkpoint handler: track macro mAP
    ckpt_handler = BestEpochCheckpoint(model, './checkpoints/', 10,
                                                          min_delta=0.005, mode='max')

    # Eager mode is great for debugging
    # Non eager graph mode is recommended for real training
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

    # Training
    for epoch in range(1, FLAGS.epochs + 1):
        for batch, (images, labels) in enumerate(train_dataset):
            images, filenames = images
            with tf.GradientTape() as tape:
                outputs = model(images, training=True)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables))

            logging.info("{}_train_{}, {}, {}".format(
                epoch, batch, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            avg_loss.update_state(total_loss)

        with train_summary_writer.as_default():
            tf.summary.scalar('Avg_loss', avg_loss.result(), step=epoch)

        all_annotations = []
        all_detections = []

        for batch, (images, labels, int_labels) in enumerate(val_dataset):
            images, filenames = images
            outputs = model(images)
            boxes, scores, classes, valid_detections = convert_yolo_output(outputs[0], outputs[1], outputs[2],
                                                                           anchors=anchors,
                                                                           anchor_masks=anchor_masks,
                                                                           num_classes=FLAGS.num_classes,
                                                                           max_boxes=FLAGS.yolo_max_boxes,
                                                                           iou_threshold=FLAGS.yolo_iou_threshold,
                                                                           score_threshold=FLAGS.yolo_score_threshold
                                                                           )
            all_annotations = create_annotations(all_annotations, int_labels, FLAGS.num_classes)
            all_detections = create_detections(all_detections, boxes, scores, classes, valid_detections,
                                               FLAGS.num_classes)
            regularization_loss = tf.reduce_sum(model.losses)
            pred_loss = []
            for output, label, loss_fn in zip(outputs, labels, loss):
                pred_loss.append(loss_fn(label, output))
            total_loss = tf.reduce_sum(pred_loss) + regularization_loss

            logging.info("{}_val_{}, {}, {}".format(
                epoch, batch, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            avg_val_loss.update_state(total_loss)

        ap_val = average_precisions(all_detections, all_annotations, FLAGS.num_classes,
                                    FLAGS.yolo_iou_threshold)

        micro_map, macro_map = calculate_map(ap_val)
        logging.info("{}, train: {}, val: {}, micro_map: {}, macro_map:{}".format(
            epoch,
            avg_loss.result().numpy(),
            avg_val_loss.result().numpy(),
            micro_map,
            macro_map))
        print('micro_map: ', micro_map)
        print('macro_map: ', micro_map)

        with val_summary_writer.as_default():
            tf.summary.scalar('Avg_loss', avg_val_loss.result(), step=epoch)
            tf.summary.scalar('Micro_mAP', micro_map, step=epoch)
            tf.summary.scalar('Macro_mAP', macro_map, step=epoch)

        ckpt_handler.on_epoch_end(epoch=epoch, current_monitor=macro_map)

        avg_loss.reset_states()
        avg_val_loss.reset_states()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
