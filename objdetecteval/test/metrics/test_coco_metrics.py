from pytest import fixture, approx

from objdetect.metrics.coco_metrics import (
    get_stats_at_annotation_level,
    conv_image_ids_to_coco,
    conv_class_labels_to_coco_cats,
    get_coco_stats,
)


@fixture
def predictions():
    # two classes
    # two images
    # two bounding box predictions for each image
    #   confidence level
    # one bounding box target ground truth for each image
    batch = {
        "predicted_class_labels": [
            [
                0,
                0,
            ],
            [1, 0],
        ],
        "predicted_class_confidences": [[0.6, 0.3], [0.6, 0.3]],
        "predicted_bboxes": [
            # image 0
            [[750.65, 276.56, 963.77, 369.68], [60, 60, 50, 50]],
            # image 1
            [[1750.65, 276.56, 1963.77, 369.68], [60, 60, 50, 50]],
        ],
        "prediction_image_ids": [0, 1],
        "target_image_ids": [0, 1],
        "target_class_labels": [
            [0],
            [1],
        ],
        "target_bboxes": [
            # image 0
            [
                [750.65, 276.56, 963.77, 369.68],
            ],
            # image 1
            [
                [750.65, 276.56, 963.77, 369.68],
            ],
        ],
    }

    expected_result = {
        "AP_all": 0.5,
        "AP_all_IOU_0_50": 0.5,
        "AP_all_IOU_0_75": 0.5,
        "AP_small": -1.0,
        "AP_medium": -1.0,
        "AP_large": 0.5,
        "AR_all_dets_1": 0.5,
        "AR_all_dets_10": 0.5,
        "AR_all": 0.5,
        "AR_small": -1.0,
        "AR_medium": -1.0,
        "AR_large": 0.5,
    }

    return (batch, expected_result)


def test_get_stats_at_annotation_level_no_bbox_func(predictions):

    batch_predictions, expected_results = predictions
    batch_predictions["conv_bbox_func"] = None
    coco_results = get_stats_at_annotation_level(**batch_predictions)
    # results should be the same as bboxes remain the same
    assert coco_results == approx(expected_results)


def test_get_stats_at_annotation_level(predictions):

    batch_predictions, expected_results = predictions
    coco_results = get_stats_at_annotation_level(**batch_predictions)

    assert coco_results == approx(expected_results)


def test_conv_class_labels_to_coco_cats():

    expected_result = [{"id": 0}, {"id": 1}, {"id": 2}]

    class_labels = [[2, 0], [1, 0], [2, 0]]

    # need unique list of categories
    coco_cats = conv_class_labels_to_coco_cats(class_labels)
    assert coco_cats == expected_result


def test_conv_image_ids_to_coco():

    expected_result = [{"id": 0}, {"id": 1}, {"id": 2}]

    # duplicate images outputs single ids
    image_ids = [0, 1, 2, 2]

    image_dict = conv_image_ids_to_coco(image_ids)
    assert image_dict == expected_result


def test_get_coco_stats(predictions):

    batch_predictions, expected_results = predictions
    batch_predictions["conv_bbox_func"] = None
    batch_predictions["include_per_class"] = False
    coco_results = get_coco_stats(**batch_predictions)

    assert coco_results["All"] == approx(expected_results)


def test_get_coco_stats_class_level(predictions):

    batch_predictions, expected_results = predictions
    batch_predictions["conv_bbox_func"] = None
    batch_predictions["include_per_class"] = True
    coco_results = get_coco_stats(**batch_predictions)

    class_0_results = {
        "AP_all": 0.9999999,
        "AP_all_IOU_0_50": 0.9999999,
        "AP_all_IOU_0_75": 0.9999999,
        "AP_large": 0.9999999,
        "AP_medium": -1.0,
        "AP_small": -1.0,
        "AR_all": 1.0,
        "AR_all_dets_1": 1.0,
        "AR_all_dets_10": 1.0,
        "AR_large": 1.0,
        "AR_medium": -1.0,
        "AR_small": -1.0,
    }
    class_1_results = {
        "AP_all": 0.0,
        "AP_all_IOU_0_50": 0.0,
        "AP_all_IOU_0_75": 0.0,
        "AP_large": 0.0,
        "AP_medium": -1.0,
        "AP_small": -1.0,
        "AR_all": 0.0,
        "AR_all_dets_1": 0.0,
        "AR_all_dets_10": 0.0,
        "AR_large": 0.0,
        "AR_medium": -1.0,
        "AR_small": -1.0,
    }

    assert coco_results[0] == approx(class_0_results)
    assert coco_results[1] == approx(class_1_results)
