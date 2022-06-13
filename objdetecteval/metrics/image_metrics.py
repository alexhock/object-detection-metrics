from objdetecteval.metrics.iou import iou
import pandas as pd
from typing import List

__all__ = [
    "get_inference_metrics",
    "summarise_inference_metrics",
    "match_preds_to_targets",
    "get_inference_metrics_from_df"
]


def get_inference_metrics_from_df(predictions_df, labels_df):

    matched_bounding_boxes = match_preds_to_targets(
        predictions_df, labels_df
    )

    return get_inference_metrics(**matched_bounding_boxes)


def get_unique_image_names(predictions_df, labels_df):
    # get unique image names from both preds and labels
    # need from both to capture images where there were no predictions
    # and images where there were predictions but no labels
    unique_preds_images = predictions_df['image_name'].unique().tolist()
    unique_label_images = labels_df['image_name'].unique().tolist()
    unique_images = sorted(list(set([*unique_preds_images, *unique_label_images])))
    return unique_images


def match_preds_to_targets(predictions_df, labels_df):

    # check for required df columns
    pred_required_columns = ['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'label', 'score']
    assert all(col in predictions_df.columns for col in pred_required_columns), \
        f"missing or different column names - should be: {pred_required_columns}"
    label_required_columns = ['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'label']
    assert all(col in labels_df.columns for col in label_required_columns), \
        f"missing or diferent column names - should be {label_required_columns}"

    image_names = []
    predicted_class_labels = []
    predicted_bboxes = []
    predicted_class_confidences = []
    target_class_labels = []
    target_bboxes = []
    image_preds = {}

    unique_images = get_unique_image_names(predictions_df, labels_df)

    # index the dataframes by the image_name
    preds_df_indexed = predictions_df.set_index('image_name')
    labels_df_indexed = labels_df.set_index('image_name')

    # loop through individual images
    for image_name in unique_images:

        # get the predictions and labels for each image
        preds = preds_df_indexed.loc[[image_name]]
        labels = labels_df_indexed.loc[[image_name]]

        # create lists for all the bounding boxes, labels and scores
        # for the image, pascal boxes
        # [[xmin, ymin, xmax, ymax], []]
        # [label, label]
        # [score, score]
        pred_image_bboxes = preds[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        pred_image_class_labels = preds['label'].values.tolist()
        pred_image_class_confs = preds['score'].values.tolist()

        # add the predictions lists for the image
        image_names.append(image_name)
        predicted_class_labels.append(pred_image_class_labels)
        predicted_class_confidences.append(pred_image_class_confs)
        predicted_bboxes.append(pred_image_bboxes)

        # create lists of the label bboxes and classes
        labels_image_bboxes = labels[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        labels_image_class_labels = labels['label'].values.tolist()

        # add the label lists for the image
        target_class_labels.append(labels_image_class_labels)
        target_bboxes.append(labels_image_bboxes)

    return {
        "image_ids": image_names,
        "predicted_class_labels": predicted_class_labels,
        "predicted_bboxes": predicted_bboxes,
        "predicted_class_confidences": predicted_class_confidences,
        "target_class_labels": target_class_labels,
        "target_bboxes": target_bboxes
    }



def calc_iou(pred_bbox, true_bboxes):
    iou_val = 0.0
    for true_bbox in true_bboxes:
        # assumes pascal
        box_iou_val = iou(pred_bbox, true_bbox)
        if box_iou_val > iou_val:
            iou_val = box_iou_val
    return iou_val


def calculate_detections(
    all_image_ids,
    all_pred_classes,
    all_pred_bboxes,
    all_pred_confs,
    all_true_classes,
    all_true_bboxes,
    do_iou_calc=True,
):
    assert len(all_image_ids) == len(all_pred_bboxes) == len(all_true_bboxes)

    # ["image_id", "class", "TP", "TN", "FP", "FN", "Confidence", "IoU"]
    detections = []

    for image_id, pred_classes, pred_boxes, pred_confs, true_classes, true_boxes in zip(
        all_image_ids,
        all_pred_classes,
        all_pred_bboxes,
        all_pred_confs,
        all_true_classes,
        all_true_bboxes,
    ):

        # loop through the predicted boxes for the image
        for pred_class, pred_box, pred_conf in zip(
            pred_classes, pred_boxes, pred_confs
        ):
            if pred_class in true_classes:
                if do_iou_calc:
                    box_iou = calc_iou(pred_box, true_boxes)
                    detections.append(
                        [image_id, pred_class, 1, 0, 0, 0, pred_conf, box_iou]
                    )
                else:
                    # true positive
                    detections.append([image_id, pred_class, 1, 0, 0, 0, pred_conf, -1])

                continue

            if pred_class not in true_classes:
                # false positive
                detections.append([image_id, pred_class, 0, 0, 1, 0, pred_conf, -1])
                continue

        # false negatives
        for true_class in true_classes:
            if true_class not in pred_classes:
                detections.append([image_id, true_class, 0, 0, 0, 1, 0, -1])

    return detections


def summarise_inference_metrics(inference_df):

    class_stats = inference_df.groupby("class")[["TP", "FP", "FN"]].sum()

    # total number for each class
    class_stats["Total"] = class_stats[["TP", "FP", "FN"]].sum(axis=1)

    class_stats["Precision"] = class_stats["TP"] / (
        class_stats["TP"] + class_stats["FP"]
    )
    class_stats["Recall"] = class_stats["TP"] / (class_stats["TP"] + class_stats["FN"])

    # remove the index creatd by the groupby so the class is a column
    class_stats = class_stats.reset_index()

    return class_stats


def get_inference_metrics(
    image_ids: List[int],
    predicted_class_labels: List[List[int]],
    predicted_bboxes: List[List[List[float]]],
    predicted_class_confidences: List[List[float]],
    target_class_labels: List[List[int]],
    target_bboxes: List[List[List[float]]],
):
    """
    Create metrics that do not include IoU. IoU is calculated but is not used to calculate precision and recall.

    Converts the outputs from the models into inference dataframes containing evaluation metrics such as
    precision and recall, and TP, FP, FN, confidence. Useful for more detailed analysis of results and plotting.

    :param image_ids: A list of image ids for each image in the order of the prediction and target lists
    :param predicted_class_labels: A list containing a list of class labels predicted per image
    :param predicted_class_confidences: A list containing a list of prediction confidence values per image
    :param predicted_bboxes: A list containing a list of bounding boxes, in Pascal VOC format, predicted per image
    :param target_class_labels: A list containing a list of ground truth class labels per image
    :param target_bboxes: A list containing a list of ground truth bounding boxes, in Pascal VOC format
    :param conv_bbox_func: A function to convert the format of incoming bboxes to pascal format, default is None
    :returns: a DataFrame of the results, and a dataframe containing precision and recall.
    """

    detections = calculate_detections(
        image_ids,
        predicted_class_labels,
        predicted_bboxes,
        predicted_class_confidences,
        target_class_labels,
        target_bboxes,
    )

    inference_df = pd.DataFrame(
        detections,
        columns=["image_id", "class", "TP", "TN", "FP", "FN", "Confidence", "IoU"],
    )

    return inference_df
