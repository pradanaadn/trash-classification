import torch
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm
from loguru import logger
from src.callback.earlystopping import EarlyStopping
from src.metric.metric import create_metrics
from src.models.evaluation import test_evaluation

def training_loop(model, criterion, optimizer, dl_train, dl_val, dl_test, wandb):
    config = wandb.config
    metrics = create_metrics(config.num_classes, config.device)

    early_stopping = EarlyStopping(
        config=config,
        mode="max",
    )

    history = []

    for epoch in range(config.epochs):
        train_loss_metric = MeanMetric().to(config.device)
        val_loss_metric = MeanMetric().to(config.device)

        log_dict = {}
        model.train()
        with tqdm(dl_train, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{config.epochs}")
            for batch in tepoch:
                try:
                    inputs = batch["image"].to(config.device)
                    labels = batch["label"].to(config.device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss_metric.update(loss.detach())
                    _, preds = torch.max(outputs, 1)

                    for metric_name, metric in metrics.items():
                        if metric_name.startswith("train"):
                            metric.update(preds, labels)

                    tepoch.set_postfix({"loss": train_loss_metric.compute().item()})
                except Exception as e:
                    logger.error(f"Error in training batch: {str(e)}")
                    continue

        model.eval()
        with torch.no_grad():
            with tqdm(dl_val, unit="batch") as tepoch:
                for batch in tepoch:
                    try:
                        inputs = batch["image"].to(config.device)
                        labels = batch["label"].to(config.device)

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss_metric.update(loss.detach())

                        _, preds = torch.max(outputs, 1)

                        for metric_name, metric in metrics.items():
                            if metric_name.startswith("val"):
                                metric.update(preds, labels)

                        tepoch.set_postfix(
                            {"val_loss": val_loss_metric.compute().item()}
                        )
                    except Exception as e:
                        logger.error(f"Error in validation batch: {str(e)}")
                        continue

        log_dict = {
            "loss": train_loss_metric.compute().item(),
            "val_loss": val_loss_metric.compute().item(),
            "epochs": epoch + 1,
        }

        for metric_name, metric in metrics.items():
            log_dict[metric_name] = metric.compute().item()
            metric.reset()

        train_loss_metric.reset()
        val_loss_metric.reset()

        history.append(log_dict)
        wandb.log(log_dict)
        logger.info(log_dict)

        early_stopping(log_dict["val_f1_macro"], model)

        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            wandb.alert(
                title="Early stopping",
                text=f"Early stopping triggered at epoch {epoch+1}",
            )
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        wandb.link_model(
            path=early_stopping.best_path, registered_model_name=config.model_name
        )
        model.load_state_dict(torch.load(early_stopping.best_path))
        test_evaluation(model, criterion, dl_test, wandb)
    except Exception as e:
        logger.error(f"Error linking model to wandb: {str(e)}")

    wandb.finish()
    return model
