import yaml
import multiprocessing as mp

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

from COCO_Dataset import get_n_classes
from COCO_DataModule import COCO_DataModule
from COCO_Module import COCO_Module


def main():
    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    NUM_WORKERS = mp.cpu_count()

    data_module = COCO_DataModule(cfg['dataset_path'], cfg['batch_size'], NUM_WORKERS)

    if cfg['checkpoint_pl']:
        model = COCO_Module.load_from_checkpoint(cfg['checkpoint_pl'])
    else:
        model = COCO_Module(num_classes=get_n_classes(cfg['dataset_path']))

    wandb.init(project=cfg['wandb_project_name'], name=cfg['wandb_run_name'])

    trainer = pl.Trainer(
        default_root_dir=cfg['checkpoint_pl_output'],
        fast_dev_run=cfg['fast_dev_run'],  # FAST_DEV_RUN,
        gpus=cfg['gpus_num'],
        logger=WandbLogger(project=cfg['wandb_project_name'], log_model=True, mode='online'),
        max_epochs=cfg['epoch_num'],
        precision=16 if cfg['gpus_num'] else 32,
        log_every_n_steps=1
    )

    trainer.fit(model, data_module)

    torch.save(
        model.model.state_dict(),
        cfg['model_state_dict_output'] + f"{cfg['wandb_run_name']}_model.pth"
    )

    wandb.finish()


if __name__ == "__main__":
    main()


