import pandas as pd
from objdetecteval.metrics import (
    image_metrics as im,
    coco_metrics as cm
)


def main():

    preds_path = './data/predictions.csv'
    labels_path = preds_path

    preds_df = pd.read_csv(preds_path)
    labels_df = pd.read_csv(labels_path)

    infer_df = im.get_inference_metrics_from_df(preds_df, labels_df)
    print(infer_df.head())
    class_summary_df = im.summarise_inference_metrics(infer_df)
    print(class_summary_df.head())

    figsize = (10, 10)
    fontsize = 24

    fig_confusion = (
        class_summary_df[["TP", "FP", "FN"]]
        .plot(kind="bar", figsize=figsize, width=1, align="center", fontsize=fontsize)
        .get_figure()
    )
    fig_confusion.savefig('./confusion.png')

    fig_pr = (
        class_summary_df[["Precision", "Recall"]]
        .plot(kind="bar", figsize=figsize, width=1, align="center", fontsize=fontsize)
        .get_figure()
    )
    fig_pr.savefig('./pr.png')

    # get coco
    res = cm.get_coco_from_dfs(preds_df, labels_df, False)
    print(res)


if __name__ == "__main__":
    main()
