import torch
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import MulticlassConfusionMatrix
from tqdm import tqdm
from loguru import logger

from src.metric.metric import create_metrics


def test_evaluation(model, criterion, dl_test, wandb):
    config = wandb.config
    metrics = create_metrics(config.num_classes, config.device, prefix_list=['test'])
    test_loss_metric = MeanMetric().to(config.device)

    log_dict = {}
    model.eval()
    with torch.no_grad():
        with tqdm(dl_test, unit="batch") as tepoch:
            for batch in tepoch:
                try:
                    inputs = batch['image'].to(config.device)
                    labels = batch['label'].to(config.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss_metric.update(loss.detach())

                    _, preds = torch.max(outputs, 1)

                    for metric_name, metric in metrics.items():
                        metric.update(preds, labels)

                    tepoch.set_postfix({"test_loss": test_loss_metric.compute().item()})
                except Exception as e:
                    logger.error(f"Error in test batch: {str(e)}")
                    continue

    log_dict = {
        'test_loss': test_loss_metric.compute().item()
    }

    for metric_name, metric in metrics.items():
        log_dict[metric_name] = metric.compute().item()
        metric.reset()

    test_loss_metric.reset()

    wandb.log(log_dict)
    logger.info(log_dict)

def confusion_matrix_test(model, dl_test, wandb, criterion, label_str):
    config = wandb.config
    model.eval()
    metric = MulticlassConfusionMatrix(num_classes=config.num_classes).to(config.device)
    with torch.no_grad():
        with tqdm(dl_test, unit="batch") as tepoch:
            for batch in tepoch:
                try:
                    inputs = batch['image'].to(config.device)
                    labels = batch['label'].to(config.device)


                    outputs = model(inputs)


                    _, preds = torch.max(outputs, 1)


                    metric.update(preds, labels)


                except Exception as e:
                    logger.error(f"Error in test batch: {str(e)}")
                    continue


    confusion_matrix = metric.compute().cpu().numpy()
    print("Confusion Matrix:\n", confusion_matrix)



    return metric



def metric_to_md_table(metric:dict):
    markdown_table = "| Key | Value |\n|-----|-------|\n"
    for key, value in metric.items():
        markdown_table += f"| {key} | {value} |\n"
    with open("export/metric.md", "w") as file:
        file.write(markdown_table)
    