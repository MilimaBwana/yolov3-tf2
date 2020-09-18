from absl import logging
import numpy as np
import tensorflow as tf
import cv2
import os
import shutil

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

YOLOV3_TINY_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
]


def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    if tiny:
        layers = YOLOV3_TINY_LAYER_LIST
    else:
        layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def create_annotations(all_annotations, int_labels, num_classes):
    """
    Adds annotations from int_labels to all_annotations.
    All_annotations is a list of annotations per class over all images.
    @param all_annotations: list to append to. shape (num_images, num_classes, x), where x is the detections in the image
        assigned to this class.
    @param int_labels: list of groundtruth int labels. shape (batch_size, max_detections, 5)
    @param num_classes: number of classes and thus length of each list per image
    """

    # Split among batch dimension
    label_list = tf.split(int_labels, tf.shape(int_labels)[0].numpy(), axis=0)

    for label in label_list:
        # Filter padding. Padding was applied if width and height of bounding box = 0
        non_padded_labels = tf.gather_nd(label, tf.where(
            tf.math.logical_and(tf.not_equal(label[:, :, 2], 0), tf.not_equal(label[:, :, 3], 0))))
        label_boxes = tf.concat(tf.split(non_padded_labels, 5, axis=-1)[:4], axis=-1)
        label_classes = tf.squeeze(tf.split(non_padded_labels, 5, axis=-1)[-1:])

        tmp = [None for i in range(num_classes)]

        for c in range(num_classes):
            tmp[c] = tf.gather_nd(label_boxes, tf.where(tf.keras.backend.flatten(label_classes) == c)).numpy()

        all_annotations.append(tmp)

    return all_annotations


def create_detections(all_detections, boxes, scores, classes, valid_detections, num_classes):
    """
    Adds detections, given from boxes, scores and classes,  to all_detections.
    All_detections is a list of detections per class over all images.
    @param all_detections: list to append to. shape (num_images, num_classes, x), where x is the detections in the image
        assigned to this class
    @param boxes: list of groundtruth int labels. shape (batch_size, max_detections, 4)
    @param scores: score for the corresponding list. shape (batch_size, max_detections, 1)
    @param classes: assigned class for the corresponding list. shape (batch_size, max_detections, 1)
    @param valid_detections: number of valid detections for each image. shape (batch_size,)
    @param num_classes: number of classes and thus length of each list per image
    """
    boxes_list = tf.split(boxes, tf.shape(boxes)[0].numpy(), axis=0)
    scores_list = tf.split(scores, tf.shape(scores)[0].numpy(), axis=0)
    classes_list = tf.split(classes, tf.shape(classes)[0].numpy(), axis=0)

    for i in range(len(boxes_list)):
        # remove first dimension and non-valid detections
        boxes = tf.reshape(boxes_list[i], tf.shape(boxes_list[i])[1:])[:valid_detections[i]]
        scores = tf.reshape(scores_list[i],
                            tf.shape(scores_list[i])[1:])[:valid_detections[i]]  # Scores are already sorted in descending order
        classes = tf.reshape(classes_list[i], tf.shape(classes_list[i])[1:])[:valid_detections[i]]

        tmp = [None for i in range(num_classes)]

        c: int
        for c in range(num_classes):
            # List of predicted boxes with corresponding score
            tmp[c] = [tf.gather_nd(boxes, tf.where(classes == c)).numpy(),
                      tf.gather_nd(scores, tf.where(classes == c)).numpy()]

        all_detections.append(tmp)

    return all_detections


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)


def clear_directory(folder, clear_subdirectories=False):
    """ Deletes every file in the given folder.
    If clear_subdirectories, subdirectories and their files are deleted, too."""
    if os.path.exists(folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path) and clear_subdirectories:
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
