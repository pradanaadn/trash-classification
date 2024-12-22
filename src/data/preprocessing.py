from typing import Tuple
from loguru import logger
import torch
from torchvision.transforms import v2
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from src.config import RANDOM_SEED


def split_dataset(
    dataset: (DatasetDict | Dataset | IterableDatasetDict | IterableDataset),
    split: Tuple[float, float, float] = None,
):
    tolerance = 1e-10

    if split is not None and abs(sum(split) - 1.0) > tolerance:
        logger.error(f"The sum of the split : {split} is not equal to one")
        return None
    train_split, val_split, test_split = split or (0.8, 0.1, 0.1)
    data = dataset.train_test_split(
        train_size=train_split, shuffle=True, seed=RANDOM_SEED
    )
    data_train = data["train"]

    data_val_test = data["test"].train_test_split(
        train_size=val_split / (val_split + test_split), seed=RANDOM_SEED
    )

    data_val = data_val_test["train"]
    data_test = data_val_test["test"]

    return data_train, data_val, data_test


def preprocessing(data_train, data_val, data_test, image_size=(224, 224)):
    if isinstance(image_size, int):
        image_sizes = (image_size, image_size)
    else:
        image_sizes = image_size
    transform_train = v2.Compose(
        [
            v2.Resize(size=image_sizes),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # v2.RandomHorizontalFlip(p=0.5),
            # v2.RandomVerticalFlip(p=0.5),
        ]
    )
    transform_val = v2.Compose(
        [
            v2.Resize(size=image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    data_train_transform = data_train.with_transform(transform_train)
    data_val_transform = data_val.with_transform(transform_val)
    data_test_transform = data_test.with_transform(transform_val)

    return data_train_transform, data_val_transform, data_test_transform


def data_loader(
    data_train,
    data_val,
    data_test,
    batch_size_train,
    batch_size_val,
    batch_size_test,
    num_worker=None,
):
    num_workers = num_worker or 0
    dl_train = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size_train, num_workers=num_workers, shuffle=True
    )
    dl_val = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size_val, num_workers=num_workers
    )
    dl_test = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size_test, num_workers=num_workers
    )
    return dl_train, dl_val, dl_test
