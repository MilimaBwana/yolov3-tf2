import tensorflow as tf
import math
import cv2 as cv
import numpy as np


def augment(img, y_train, params):
    augmentations = params['augmentation_techniques']

    augmentation_functions = {
        'flip_left_right': flip_left_right,
        'flip_up_down': flip_up_down,
        'rotate': rotate,
        'noise': noise,
        'flip_left_right_bboxes': flip_left_right_bboxes,
        'flip_up_down_bboxes': flip_up_down_bboxes}

    # remove class label and concat it later
    boxes = y_train[:, :4]
    classes = y_train[:, 4:]

    for augmentation in augmentations:
        # Probability of augmenting is 0.25 """
        if augmentation in augmentation_functions.keys():
            f = augmentation_functions[augmentation]
            if tf.random.uniform([], 0, 1) > 0:
                img, boxes = f(img, boxes)
            #img, boxes = tf.cond(tf.math.greater(tf.random.uniform([], 0, 1), 0.75), lambda: f(img, boxes), lambda: img,boxes)
        else:
            raise ValueError('No valid augmentation: ', augmentation)

    y_train = tf.concat([boxes, classes], axis=-1)
    return img, y_train


def noise(image, boxes):
    """ Adds random noise from a normal distribution.
    @param image: image to put noise onto.
    @param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [xmin, ymin, xmax, ymax]
    @return: the noised image.
    """
    noise = tf.random.normal(shape=tf.shape(image), mean=15, stddev=1, dtype=tf.float32)
    image = tf.add(image, noise)
    image = tf.clip_by_value(image, 0, 255)
    return image, boxes


def flip_left_right(image, boxes):
    """
    https://github.com/Ximilar-com/tf-image/blob/master/tf_image/core/bboxes/flip.py
    Flip an image and bounding boxes horizontally (left to right).
    @param image: 3-D Tensor of shape [height, width, channels]
    @param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [xmin, ymin, xmax, ymax]
    @return: image, bounding boxes
    """
    boxes = boxes * tf.constant([-1, 1, -1, 1], dtype=tf.float32) + tf.stack([1.0, 0.0, 1.0, 0.0])
    boxes = tf.stack([boxes[:, 2], boxes[:, 1], boxes[:, 0], boxes[:, 3]], axis=1)
    image = tf.image.flip_left_right(image)

    return image, boxes


def flip_up_down(image, boxes):
    """
    https://github.com/Ximilar-com/tf-image/blob/master/tf_image/core/bboxes/flip.py
    Flip an image and bounding boxes vertically (upside down).
    @param image: 3-D Tensor of shape [height, width, channels]
    @param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [xmin, ymin, xmax, ymax]
    @return: image, bounding boxes
    """
    boxes = boxes * tf.constant([1, -1, 1, -1], dtype=tf.float32) + tf.stack([0.0, 1.0, 0.0, 1.0])
    boxes = tf.stack([boxes[:, 0], boxes[:, 3], boxes[:, 2], boxes[:, 1]], axis=1)
    image = tf.image.flip_up_down(image)

    return image, boxes


def rotate(image, boxes):
    """
    Rotates the image and bounding boxes by a random degree.
    @param image: 3-D Tensor of shape [height, width, channels]
    @param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [xmin, ymin, xmax, ymax]
    """
    degree = tf.random.uniform([], -20, 20, tf.dtypes.float32)
    boxes = tf.map_fn(fn=lambda x: __rotate_single_bbox(tf.squeeze(x), tf.shape(image)[1], tf.shape(image)[0], degree),
                      elems=boxes)
    boxes = tf.stack(boxes)
    image = __rotate_img(image, tf.shape(image), -degree)
    return image, boxes


def flip_left_right_bboxes(img, boxes):
    """
    Turns each bounding box left and right with a 50 percent probability of 50 percent each.
    @param img: 3-D Tensor of shape [height, width, channels]
    @param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [xmin, ymin, xmax, ymax]
    """

    for box in boxes:
        if tf.random.uniform([], 0, 1) > 0.5:
            # Every box has probability of 0.5 to get flipped
            img = __flip_single_bbox(img, box, tf.image.flip_left_right)
    return img, boxes


def flip_up_down_bboxes(img, boxes):
    """
    Turns each bounding box up and down with a 50 percent probability of 50 percent each.
    @param img: 3-D Tensor of shape [height, width, channels]
    @param boxes: 2-D Tensor of shape (box_number, 4) containing bounding boxes in format [xmin, ymin, xmax, ymax]
    """

    for box in boxes:
        if tf.random.uniform([], 0, 1) > 0.5:
            # Every box has probability of 0.5 to get flipped
            img = __flip_single_bbox(img, box, tf.image.flip_up_down)
    return img, boxes


def __flip_single_bbox(image, bbox, op):
    """
    Flips a bbox inside an image by the given op.
    @param image: 3-D Tensor of shape [height, width, channels]
    @param bbox: 1-D Tensor of shape (4, ) containing a bounding box in format [xmin, ymin, xmax, ymax]
    @param op: tf.image.flip_up_down or tf.image.flip_left_right
    Original code from:
    https: // github.com / tensorflow / tpu / blob / c1a3ab6f7bb5d37a4cfff5ffc204d77a39e43668 / models / official / detection / utils / autoaugment_utils.py  # L505
    """

    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)
    min_x = tf.cast(image_height * bbox[0], tf.int32)
    min_y = tf.cast(image_width * bbox[1], tf.int32)
    max_x = tf.cast(image_height * bbox[2], tf.int32)
    max_y = tf.cast(image_width * bbox[3], tf.int32)
    image_height = tf.cast(image_height, tf.int32)
    image_width = tf.cast(image_width, tf.int32)

    # Clip to be sure the max values do not fall out of range.
    max_y = tf.minimum(max_y, image_height - 1)
    max_x = tf.minimum(max_x, image_width - 1)

    # Get the sub-tensor that is the image within the bounding box region.
    bbox_content = image[min_y:max_y + 1, min_x:max_x + 1, :]

    # Apply the augmentation function to the bbox portion of the image.
    augmented_bbox_content = op(bbox_content)

    # Pad the augmented_bbox_content and the mask to match the shape of original
    # image.
    augmented_bbox_content = tf.pad(augmented_bbox_content,
                                    [[min_y, (image_height - 1) - max_y],
                                     [min_x, (image_width - 1) - max_x],
                                     [0, 0]])

    # Create a mask that will be used to zero out a part of the original image.
    mask_tensor = tf.zeros_like(bbox_content)

    mask_tensor = tf.pad(mask_tensor,
                         [[min_y, (image_height - 1) - max_y],
                          [min_x, (image_width - 1) - max_x],
                          [0, 0]],
                         constant_values=1)
    # Replace the old bbox content with the new augmented content.
    image = image * mask_tensor + augmented_bbox_content
    return image


def __rotate_img(img, input_shape, degree):
    """ Rotates the image randomly between -45 and 45 degrees and fills the borders with black pixels.
    @param img: image to rotate.
    @param input_shape: shape of the outputted image.
    @return: the rotated image.
    """

    def __cv2_rotate(image, deg):
        """ Rotates the image by deg. """
        num_rows, num_cols = image.shape[:2]

        rotation_matrix = cv.getRotationMatrix2D((num_cols / 2, num_rows / 2), deg, 1)
        image = cv.warpAffine(np.float32(image), rotation_matrix, (num_cols, num_rows))
        # In case of only one channel, warpAffine removes channel dimension."""
        return image

    img = tf.py_function(func=__cv2_rotate, inp=[img, degree], Tout=tf.float32)
    # In case of only one channel, _cv2_rotate removes channel dimension, which needs to be added afterwards. """
    if input_shape[2] == 1:
        img = tf.expand_dims(img, axis=-1)
    return img


def __rotate_single_bbox(bbox, image_height, image_width, degrees):
    """Rotates the bbox coordinated by degrees.
    https://github.com/tensorflow/tpu/blob/c1a3ab6f7bb5d37a4cfff5ffc204d77a39e43668/models/official/detection/utils/autoaugment_utils.py#L437
    Args:
      bbox: 1D Tensor that has 4 elements (min_y, min_x, max_y, max_x)
        of type float that represents the normalized coordinates between 0 and 1.
      image_height: Int, height of the image.
      image_width: Int, height of the image.
      degrees: Float, a scalar angle in degrees to rotate all images by. If
        degrees is positive the image will be rotated clockwise otherwise it will
        be rotated counterclockwise.
    Returns:
      A tensor of the same shape as bbox, but now with the rotated coordinates.
    """
    image_height, image_width = (
        tf.cast(image_height, tf.float32), tf.cast(image_width, tf.float32))

    # Convert from degrees to radians.
    degrees_to_radians = math.pi / 180.0
    radians = degrees * degrees_to_radians

    # Translate the bbox to the center of the image and turn the normalized 0-1
    # coordinates to absolute pixel locations.
    # Y coordinates are made negative as the y axis of images goes down with
    # increasing pixel values, so we negate to make sure x axis and y axis points
    # are in the traditionally positive direction.
    min_y = -tf.cast(image_height * (bbox[0] - 0.5), tf.int32)
    min_x = tf.cast(image_width * (bbox[1] - 0.5), tf.int32)
    max_y = -tf.cast(image_height * (bbox[2] - 0.5), tf.int32)
    max_x = tf.cast(image_width * (bbox[3] - 0.5), tf.int32)
    coordinates = tf.stack(
        [[min_y, min_x], [min_y, max_x], [max_y, min_x], [max_y, max_x]])
    coordinates = tf.cast(coordinates, tf.float32)
    # Rotate the coordinates according to the rotation matrix clockwise if
    # radians is positive, else negative
    rotation_matrix = tf.stack(
        [[tf.cos(radians), tf.sin(radians)],
         [-tf.sin(radians), tf.cos(radians)]])
    new_coords = tf.cast(
        tf.matmul(rotation_matrix, tf.transpose(coordinates)), tf.int32)
    # Find min/max values and convert them back to normalized 0-1 floats.
    min_y = -(tf.cast(tf.reduce_max(new_coords[0, :]), tf.float32) / image_height - 0.5)
    min_x = tf.cast(tf.reduce_min(new_coords[1, :]), tf.float32) / image_width + 0.5
    max_y = -(tf.cast(tf.reduce_min(new_coords[0, :]), tf.float32) / image_height - 0.5)
    max_x = tf.cast(tf.reduce_max(new_coords[1, :]), tf.float32) / image_width + 0.5

    # Clip the bboxes to be sure the fall between [0, 1].
    min_y, min_x, max_y, max_x = __clip_bbox(min_y, min_x, max_y, max_x)
    min_y, min_x, max_y, max_x = __check_bbox_area(min_y, min_x, max_y, max_x)

    return tf.stack([min_y, min_x, max_y, max_x])


def __check_bbox_area(min_y, min_x, max_y, max_x, delta=0.05):
    """Adjusts bbox coordinates to make sure the area is > 0.
    Args:
      min_y: Normalized bbox coordinate of type float between 0 and 1.
      min_x: Normalized bbox coordinate of type float between 0 and 1.
      max_y: Normalized bbox coordinate of type float between 0 and 1.
      max_x: Normalized bbox coordinate of type float between 0 and 1.
      delta: Float, this is used to create a gap of size 2 * delta between
        bbox min/max coordinates that are the same on the boundary.
        This prevents the bbox from having an area of zero.
    Returns:
      Tuple of new bbox coordinates between 0 and 1 that will now have a
      guaranteed area > 0.
    """
    height = max_y - min_y
    width = max_x - min_x

    def __adjust_bbox_boundaries(min_coord, max_coord):
        # Make sure max is never 0 and min is never 1.
        max_coord = tf.maximum(max_coord, 0.0 + delta)
        min_coord = tf.minimum(min_coord, 1.0 - delta)
        return min_coord, max_coord

    min_y, max_y = tf.cond(tf.equal(height, 0.0),
                           lambda: __adjust_bbox_boundaries(min_y, max_y),
                           lambda: (min_y, max_y))
    min_x, max_x = tf.cond(tf.equal(width, 0.0),
                           lambda: __adjust_bbox_boundaries(min_x, max_x),
                           lambda: (min_x, max_x))
    return min_y, min_x, max_y, max_x


def __clip_bbox(min_y, min_x, max_y, max_x):
    """Clip bounding box coordinates between 0 and 1.
    Args:
      min_y: Normalized bbox coordinate of type float between 0 and 1.
      min_x: Normalized bbox coordinate of type float between 0 and 1.
      max_y: Normalized bbox coordinate of type float between 0 and 1.
      max_x: Normalized bbox coordinate of type float between 0 and 1.
    Returns:
      Clipped coordinate values between 0 and 1.
    """
    min_y = tf.clip_by_value(min_y, 0.0, 1.0)
    min_x = tf.clip_by_value(min_x, 0.0, 1.0)
    max_y = tf.clip_by_value(max_y, 0.0, 1.0)
    max_x = tf.clip_by_value(max_x, 0.0, 1.0)
    return min_y, min_x, max_y, max_x