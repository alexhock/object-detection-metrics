import numpy as np


def boxes_intersect(box_a, box_b):
    if box_a[0] > box_b[2]:
        return False  # boxA is right of boxB
    if box_b[0] > box_a[2]:
        return False  # boxA is left of boxB
    if box_a[3] < box_b[1]:
        return False  # boxA is above boxB
    if box_a[1] > box_b[3]:
        return False  # boxA is below boxB
    return True


def get_intersection_area(box_a, box_b):
    if boxes_intersect(box_a, box_b) is False:
        return 0
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    # intersection area
    return (xB - xA + 1) * (yB - yA + 1)


def get_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def get_union_areas(box_a, box_b, interArea=None):
    area_A = get_area(box_a)
    area_B = get_area(box_b)
    if interArea is None:
        interArea = get_intersection_area(box_a, box_b)
    return float(area_A + area_B - interArea)


def iou(box_a, box_b):
    # if boxes dont intersect
    if boxes_intersect(box_a, box_b) is False:
        return 0
    inter_area = get_intersection_area(box_a, box_b)
    union = get_union_areas(box_a, box_b, interArea=inter_area)
    # intersection over union
    iou = inter_area / union
    assert iou >= 0
    return iou
