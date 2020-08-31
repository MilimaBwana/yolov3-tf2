import tensorflow as tf
import math
from .augmentation import augment


def load_tf_record(mode, params, anchors, anchor_masks):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        params['dataset'].name_classes, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_dataset = tf.data.TFRecordDataset(params['dataset'].tf_record_train)
        # Tests
        #for x in train_dataset:
        #    parse_example(x, class_table, params, mode)
        train_dataset = train_dataset.map(lambda x: parse_example(x, class_table, params, mode))
        train_dataset = train_dataset.shuffle(buffer_size=512)
        train_dataset = train_dataset.batch(params['batch_size'])
        # Tests
        # for x, y in train_dataset:
        #    transform_images(x, params['img_size'])
        #    transform_targets(y, anchors, anchor_masks, params['img_size'])
        train_dataset = train_dataset.map(lambda x, y: (
            __transform_images(x, params['img_size']),
            __transform_targets(y, anchors, anchor_masks, params['img_size'])))
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        return train_dataset

    elif mode == tf.estimator.ModeKeys.EVAL:
        val_dataset = tf.data.TFRecordDataset(params['dataset'].tf_record_val)
        val_dataset = val_dataset.map(lambda x: parse_example(x, class_table, params, mode))
        val_dataset = val_dataset.batch(params['batch_size'])
        val_dataset = val_dataset.map(lambda x, y: (
            __transform_images(x, params['img_size']),
            __transform_targets(y, anchors, anchor_masks, params['img_size']),
            y))
        return val_dataset
    elif mode == tf.estimator.ModeKeys.PREDICT:
        test_dataset = tf.data.TFRecordDataset(params['dataset'].tf_record_test)
        test_dataset = test_dataset.map(lambda x: parse_example(x, class_table, params, mode))
        test_dataset = test_dataset.map(lambda x, y: (
            __transform_images(x, params['img_size']), y))
        return test_dataset
    else:
        raise ValueError('No valid mode.')


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def __transform_targets(y_train, anchors, anchor_masks, size):
    """
    Bounds targets to the nearest anchor box.
    """
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
                   tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def __transform_images(x_train, size):
    """
    Resizes image to target size and normalizes it.
    @param x_train: tuple of (image, filename)
    @param size: target size
    @return: resized image and filename of image.
    """
    img = x_train[0]
    filename = x_train[1]
    img = tf.image.resize(img, (size, size))
    img = img / 255
    return img, filename


# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# Commented out fields are not required in our project
IMAGE_FEATURE_MAP = {
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}


def parse_example(serialized_example, class_table, params, mode):
    """ Parses a image with bounding boxes, filename and class label.
    @param serialized_example: one serialized example out of a tf record.
    @param class_table: tf.lookup.StaticHashTable with class text (key) and label (value)
    @param params: dictionary containing important hyperparameters. Must contain
            'max_boxes', 'img_size', 'augmentation' as a key.
            If 'augmentation', the 'augmentation_techniques' key is also needed.
    @return a parsed (x_train, y_train)-tuple. x_train contains (img, filename)-tuple
    """
    x = tf.io.parse_single_example(serialized_example, IMAGE_FEATURE_MAP)
    img = tf.image.decode_png(x['image/encoded'], channels=3)
    filename = tf.cast(x['image/filename'], tf.string)
    img = tf.cast(img, tf.float32)

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)

    if mode == tf.estimator.ModeKeys.TRAIN and params['augmentation']:
        img, y_train = augment(img, y_train, params)

    img = tf.image.resize(img, (params['img_size'], params['img_size']))
    paddings = [[0, params['max_boxes'] - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    x_train = (img, filename)

    return x_train, y_train
