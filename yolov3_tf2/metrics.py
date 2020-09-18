import numpy as np
import tensorflow as tf


def broadcast_iou(box_1, box_2):
    """
    Given two arrays `a` and `b` where each row contains a bounding
    box defined as a list of four numbers:
    [x1,y1,x2,y2]
    where:
        x1,y1 represent the upper left corner
        x2,y2 represent the lower right corner
    @param box_1: (..., (x1, y1, x2, y2))
    @param box_2: (N, (x1, y1, x2, y2))
    @return  the Intersection of Union scores for each corresponding
    pair of boxes.
    """

    # broadcast boxes
    box_1 = np.expand_dims(box_1, -2)
    box_2 = np.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(np.shape(box_1), np.shape(box_2)).numpy()
    box_1 = np.broadcast_to(box_1, new_shape)
    box_2 = np.broadcast_to(box_2, new_shape)

    int_w = np.maximum(np.minimum(box_1[..., 2], box_2[..., 2]) -
                       np.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = np.maximum(np.minimum(box_1[..., 3], box_2[..., 3]) -
                       np.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    iou = int_area / (box_1_area + box_2_area - int_area)
    return iou


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    @param recall:    The recall curve (list).
    @param precision: The precision curve (list).
    @return: The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def average_precisions(all_detections, all_annotations, num_classes, iou_threshold):
    """
    Calculates the average precision for each class.
    @param all_detections: List of detections
    @param all_annotations: List of annotations
    @param num_classes: number of classes
    @param iou_threshold: similarity threshold that marks a true positive.
    @return dictionary of average precisions per class
    """

    average_precisions = {}

    # process detections and annotations
    for label in range(num_classes):

        # TODO: continue if no labels

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(all_annotations)):
            detections = all_detections[i][label]
            detection_bb, detection_scores = detections
            #
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for idx, d in enumerate(detection_bb):

                scores = np.append(scores, detection_scores[idx])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = broadcast_iou(np.expand_dims(d, axis=0), annotations)  # Overlaps gives back IoU Value
                assigned_annotation = np.squeeze(np.argmax(overlaps, axis=-1))
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


def calculate_map(aps):
    """
    Prints the average precision for each class and calculates micro mAP and macro mAP.
    @param aps: average precision for each class
    @param dictionary: mapping between label anc class name
    @return: micro mAP and macro mAP
    """
    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations) in aps.items():
        total_instances.append(num_annotations)
        precisions.append(average_precision)

    if sum(total_instances) == 0:
        print('No test instances found.')
        return 0, 0

    micro_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
    macro_ap = sum(precisions) / sum(x > 0 for x in total_instances)
    print('Micro mAP : {:.4f}'.format(micro_ap))
    print('Macro mAP: {:.4f}'.format(macro_ap))

    return micro_ap, macro_ap
