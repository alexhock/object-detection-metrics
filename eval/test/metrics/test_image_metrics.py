import pandas as pd
from pandas._testing import assert_frame_equal
from pytest import fixture, approx

from eval.metrics.image_metrics import (
    get_inference_metrics,
    summarise_inference_metrics,
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

    expected_inference_df = pd.DataFrame(
        {
            "image_id": [0, 0, 1, 1],
            "class": [0, 0, 1, 0],
            "TP": [1, 1, 1, 0],
            "TN": [0, 0, 0, 0],
            "FP": [0, 0, 0, 1],
            "FN": [0, 0, 0, 0],
            "Confidence": [0.6, 0.3, 0.6, 0.3],
            "IoU": [1.0, 0.0, 0.0, -1.0],
        }
    )

    expected_class_metrics_df = pd.DataFrame(
        {
            "class": [0, 1],
            "TP": [2, 1],
            "FP": [1, 0],
            "FN": [0, 0],
            "Total": [3, 1],
            "Precision": [0.666667, 1.0],
            "Recall": [1.0, 1.0],
        }
    )

    return (batch, expected_inference_df, expected_class_metrics_df)


def test_get_inference_metrics(predictions):

    preds, expected_inference_df, _ = predictions

    inference_df = get_inference_metrics(
        image_ids=preds["prediction_image_ids"],
        predicted_class_labels=preds["predicted_class_labels"],
        predicted_class_confidences=preds["predicted_class_confidences"],
        predicted_bboxes=preds["predicted_bboxes"],
        target_class_labels=preds["target_class_labels"],
        target_bboxes=preds["target_bboxes"],
    )

    assert_frame_equal(expected_inference_df, inference_df)


def test_summarise_inference_metrics(predictions):

    preds, _, expected_class_metrics_df = predictions

    inference_df = get_inference_metrics(
        image_ids=preds["prediction_image_ids"],
        predicted_class_labels=preds["predicted_class_labels"],
        predicted_class_confidences=preds["predicted_class_confidences"],
        predicted_bboxes=preds["predicted_bboxes"],
        target_class_labels=preds["target_class_labels"],
        target_bboxes=preds["target_bboxes"],
    )

    summary_metrics_df = summarise_inference_metrics(inference_df)

    assert_frame_equal(
        expected_class_metrics_df, summary_metrics_df, check_exact=False, rtol=1e-4
    )
