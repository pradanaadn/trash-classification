from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)


def create_metrics(num_classes, device, prefix_list=None):
    prefixs = prefix_list or ["train", "val"]
    metrics = {}
    for prefix in prefixs:
        for average in ["macro", "micro", "weighted"]:
            metrics[f"{prefix}_accuracy_{average}"] = MulticlassAccuracy(
                num_classes=num_classes, average=average
            ).to(device)
            metrics[f"{prefix}_precision_{average}"] = MulticlassPrecision(
                num_classes=num_classes, average=average
            ).to(device)
            metrics[f"{prefix}_recall_{average}"] = MulticlassRecall(
                num_classes=num_classes, average=average
            ).to(device)
            metrics[f"{prefix}_f1_{average}"] = MulticlassF1Score(
                num_classes=num_classes, average=average
            ).to(device)
    return metrics
