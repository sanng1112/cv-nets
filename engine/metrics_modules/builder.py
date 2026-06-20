import torchmetrics

def build_metrics(task_type: str, num_classes: int = None):
    """
    Tự động xây dựng bộ đánh giá siêu chi tiết dựa trên loại bài toán.
    """
    if not task_type:
        return None
        
    task_type = task_type.lower()
    
    if task_type == "classification":
        return torchmetrics.MetricCollection({
            "Acc": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
            "Precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro'),
            "Recall": torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro'),
            "F1": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro')
        })
    elif task_type == "segmentation":
        return torchmetrics.MetricCollection({
            "mIoU": torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes),
            "Dice": torchmetrics.Dice(num_classes=num_classes)
        })
    elif task_type == "regression":
        return torchmetrics.MetricCollection({
            "MSE": torchmetrics.MeanSquaredError(),
            "MAE": torchmetrics.MeanAbsoluteError(),
            "R2": torchmetrics.R2Score()
        })
    elif task_type == "detection":
        # Detection mAP yêu cầu format boxes/labels cụ thể
        return torchmetrics.MetricCollection({
            "mAP": torchmetrics.detection.MeanAveragePrecision()
        })
    else:
        return None
