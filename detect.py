import time
from absl import app, flags, logging
from absl.flags import FLAGS
from pathlib import Path
import tensorflow as tf

from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks,
    convert_yolo_output
)
from yolov3_tf2.dataset import __transform_images
from yolov3_tf2.detection_visualization import draw_detections
from yolov3_tf2.utils import create_detections, create_annotations, clear_directory
from yolov3_tf2.metrics import average_precisions, calculate_map
import yolov3_tf2.dataset as dataset

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('log_dir', './logs/predict', 'path to log dir')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        anchor_masks = yolo_tiny_anchor_masks
        anchors = yolo_tiny_anchors

        model = YoloV3Tiny(FLAGS.size, training=False, channels=3, anchors=anchors,
                           masks=anchor_masks, classes=FLAGS.num_classes)
    else:
        anchor_masks = yolo_anchor_masks
        anchors = yolo_anchors
        model = YoloV3(FLAGS.size, channels=3, anchors=anchors,
                       masks=anchor_masks, training=False, classes=FLAGS.num_classes)

    # Load pretrained model.
    checkpoint = tf.train.Checkpoint(net=model)
    if FLAGS.weights:
        checkpoint_dir = FLAGS.weights
    else:
        checkpoint_dir = 'checkpoints/'
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    status.assert_existing_objects_matched()
    status.expect_partial()
    logging.info('weights loaded')

    class_dictionary = {}
    with open(FLAGS.classes) as f:
        for idx, val in enumerate(f):
            class_dictionary[idx] = val.strip('\n')

    if FLAGS.tfrecord:
        test_dataset = dataset.load_tf_record(tfrecord=FLAGS.tfrecord, mode=tf.estimator.ModeKeys.PREDICT,
                                              class_file=FLAGS.classes, anchors=anchors,
                                              anchor_masks=anchor_masks, batch_size=1,
                                              max_detections=FLAGS.yolo_max_boxes, size=FLAGS.size,
                                              augmentation=False)

        all_annotations = []
        all_detections = []

        predict_summary_writer = tf.summary.create_file_writer(FLAGS.log_dir)
        clear_directory(FLAGS.log_dir, clear_subdirectories=True)
        Path(FLAGS.log_dir + '/images').mkdir(parents=True, exist_ok=True)

        for image, int_label in test_dataset:
            image, filename = image
            image = tf.expand_dims(image, axis=0)
            int_label = tf.expand_dims(int_label, axis=0)
            # Outputs shape [(batch, 13, 13, 3, 5 + classes), (batch, 26, 26, 3, 5 + classes),
            # (batch, 52, 52, 3, 5 + classes)]
            outputs = model(image)
            boxes, scores, classes, valid_detections = convert_yolo_output(outputs[0], outputs[1], outputs[2],
                                                                           anchors,
                                                                           yolo_anchor_masks,
                                                                           FLAGS.num_classes,
                                                                           FLAGS.yolo_max_boxes,
                                                                           FLAGS.yolo_iou_threshold,
                                                                           FLAGS.yolo_score_threshold
                                                                           )

            # Remove non valid detections
            boxes = boxes[:, :tf.squeeze(valid_detections)]
            scores = scores[:, :tf.squeeze(valid_detections)]
            classes = classes[:, :tf.squeeze(valid_detections)]

            draw_detections(image, boxes, scores, classes, filename,
                            FLAGS.log_dir + '/images',
                            class_dictionary, FLAGS.yolo_score_threshold)

            # Save annotations and detections in list
            all_annotations = create_annotations(all_annotations, int_label, FLAGS.num_classes)
            all_detections = create_detections(all_detections, boxes, scores, classes, valid_detections,
                                               FLAGS.num_classes)

        ap_predict = average_precisions(all_detections, all_annotations, FLAGS.num_classes,
                                        FLAGS.yolo_iou_threshold)

        micro_map, macro_map = calculate_map(ap_predict)
        print('micro_map: ', micro_map)
        print('macro_map: ', micro_map)

        with predict_summary_writer.as_default():
            tf.summary.scalar('Micro mAP', micro_map, step=1)
            tf.summary.scalar('Macro mAP', macro_map, step=1)

    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)
        img = tf.expand_dims(img_raw, 0)
        img, filename = __transform_images(img, FLAGS.size)
        outputs = model(img)
        boxes, scores, classes, valid_detections = convert_yolo_output(outputs[0], outputs[1], outputs[2],
                                                                       anchors=anchors,
                                                                       anchor_masks=anchor_masks,
                                                                       num_classes=FLAGS.num_classes,
                                                                       max_boxes=FLAGS.yolo_max_boxes,
                                                                       iou_threshold=FLAGS.yolo_iou_threshold,
                                                                       score_threshold=FLAGS.yolo_score_threshold
                                                                       )
        # TODO: show detection on image



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
