import cv2
import tensorflow as tf
import numpy as np
from pathlib import Path
import os


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.
    @param image     : The image to draw on.
    @param box       : A list of 4 elements (x1, y1, x2, y2).
    @param color     : The color of the box.
    @param thickness : The thickness of the lines to draw a box with.
    @return image with drawn box.
    """
    b = np.array(box).astype(int)
    image = image.astype(np.uint8)
    image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA) # RGB color
    return image


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.
    @param image   : The image to draw on.
    @param box     : A list of 4 elements (x1, y1, x2, y2).
    @param caption : String containing the text to draw.
    @return image with drawn caption.
    """
    b = np.array(box).astype(int)
    image = cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 2)
    image = cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)
    return image


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.
    @param image     : The image to draw on.
    @param boxes     : A [N, 4] matrix (x1, y1, x2, y2).
    @param color     : The color of the boxes.
    @param thickness : The thickness of the lines to draw boxes with.
    @return image with drawn boxes
    """
    for b in boxes:
        image = draw_box(image, b, color, thickness=thickness)

    return image


def draw_detections(image, boxes, scores, classes, filename, save_dir, label_to_name, score_threshold=0.20):
    """ Draws detections in an image.
    @param image: The image to draw on. Image size is determined by input size for network.
    @param boxes: A [N, 4] matrix (x1, y1, x2, y2).
    @param scores: A list of N classification scores.
    @param classes: A list of N labels.
    @param filename: filename of the image.
    @param save_dir: directory to save image in.
    @param label_to_name: (optional) Functor for mapping a label to a name.
    @param score_threshold: Threshold used for determining what detections to draw.
    """

    # Remove batch dimension
    image = tf.reshape(image, tf.shape(image)[1:]).numpy() * 255
    boxes = tf.reshape(boxes, tf.shape(boxes)[1:]).numpy()
    classes = tf.reshape(classes, tf.shape(classes)[1:]).numpy()

    if not boxes.shape[0] == 0:
        # if boxes are detected
        save_dir = os.path.join(save_dir, Path(filename.numpy().decode('utf-8')).stem + '_detections.png')
        # Convert to absolute coordinates
        boxes[:, 0] *= image.shape[0]
        boxes[:, 1] *= image.shape[1]
        boxes[:, 2] *= image.shape[0]
        boxes[:, 3] *= image.shape[1]

        selection = np.where(scores > score_threshold)[1]

        scores = tf.reshape(scores, tf.shape(scores)[1:]).numpy()

        for i in selection:
            image = draw_box(image, boxes[i, :], color=(0, 255, 0))

            # draw labels
            caption = (label_to_name[classes[i]] if label_to_name else classes[i]) + ': {0:.2f}'.format(scores[i])
            image = draw_caption(image, boxes[i, :], caption)

        cv2.imwrite(save_dir, image)

