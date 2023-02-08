import pytorch_lightning as pl
from COCO_Dataset import COCO_Dataset
from torch.utils.data.dataloader import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.Resize(600, 600),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.1),
            A.ColorJitter(p=0.1),
            # A.Normalize(mean=[0.4784, 0.4453, 0.3952], std=[0.2655, 0.2599, 0.2674]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    else:
        transform = A.Compose([
            A.Resize(600, 600),
            # A.Normalize(mean=[0.4784, 0.4453, 0.3952], std=[0.2655, 0.2599, 0.2674]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco'))
    return transform


class COCO_DataModule(pl.LightningDataModule):

    def __init__(self, dataset_path, batch_size, num_workers):
        super().__init__()

        self.save_hyperparameters()

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_dataset = COCO_Dataset(self.dataset_path, split='train', transforms=get_transforms(True))
        val_dataset = COCO_Dataset(self.dataset_path, split='valid', transforms=get_transforms(True))

        # Split
        # len_total = len(train_dataset)
        # len_train = int(0.8 * len_total)
        # indices = torch.randperm(len_total).tolist()
        # train_dataset = torch.utils.data.Subset(train_dataset, indices[:len_train])
        # val_dataset = torch.utils.data.Subset(val_dataset, indices[len_train:])

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )
