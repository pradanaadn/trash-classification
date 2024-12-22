import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import wandb
from datasets import load_dataset
from src.data.preprocessing import data_loader, split_dataset, preprocessing
from src.models.training import training_loop
from src.models.evaluation import confusion_matrix_test
from src.models.model import TrashMobileNet


def main():
    data_loading = load_dataset("garythung/trashnet", split="train")
    label = data_loading.features["label"].names

    runs = wandb.init(
        project="ci_cd_Trash_Classification",
        name="ci_cd_Trash_Classification",
        config={
            "learning_rate": 0.001,
            "repo_name":"ci_cd_Trash_Classification",
            "architecture": "Trashmobilenet-1",
            "dataset": "garythung/trashnet",
            "epochs": 1,
            "device": "cpu",
            "early_stopping_patience": 3,
            "train_batch_size": 16,
            "val_batch_size": 16,
            "test_batch_size": 16,
            "num_workers": 2,
            "num_classes": len(label),
            "optimizer": "Adam",
            "model_name": "Trashmobilenet-v1",
        },
        settings=wandb.Settings(init_timeout=120),
        reinit=True,
    )

    train_set, val_set, test_set = split_dataset(data_loading)
    train_tf, val_tf, test_tf = preprocessing(train_set, val_set, test_set)
    train_dl, val_dl, test_dl = data_loader(
        train_tf,
        val_tf,
        test_tf,
        runs.config.train_batch_size,
        runs.config.val_batch_size,
        runs.config.test_batch_size,
        runs.config.num_workers
    )
    model_init = TrashMobileNet(num_classes=runs.config.num_classes).to(runs.config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_init.parameters(), lr=runs.config.learning_rate)
    model = training_loop(model_init, criterion, optimizer,train_dl, val_dl, test_dl, runs )
    cmt = confusion_matrix_test(model, test_dl, runs, criterion, label)
    fig_, ax_ = cmt.plot(cmap=plt.cm.Blues, add_text = True, labels = label )
    ax_.set_title(f"{runs.config.architecture} Confusion Matrix")
    fig_.savefig("export/confusion_matrix.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
