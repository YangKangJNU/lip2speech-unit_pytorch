
import argparse
import json
import os

import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HYDRA_FULL_ERROR'] = '1'
from src.trainer_s1 import (
    VADataset,
    Trainer,
)
import hydra
from omegaconf import DictConfig
from src.models.model import L2SUnit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--lr', default=1e-4)
    args = parser.parse_args()
    return args

@hydra.main(version_base=None, config_path="configs/v1", config_name="default.yaml")
def main(cfg: DictConfig):
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    train_dataset = VADataset('train', cfg.data)
    val_dataset = VADataset('val', cfg.data)

    model = L2SUnit(cfg.encoder)

    if args.lr is not None:
        lr_train = args.lr
    else:
        lr_train = cfg.train.lr
    trainer = Trainer(
        model=model,
        num_warmup_steps=cfg.train.num_warmup_steps,
        lr=lr_train,
        grad_accumulation_steps = 1,
        tensorboard_log_dir=args.output_dir,
        checkpoint_path = os.path.join(args.output_dir, 'model.pt'),
        log_file = os.path.join(args.output_dir, 'logs.txt')
    )

    trainer.train(cfg.train,
                   train_dataset,
                   val_dataset = val_dataset,
                   ckpt_path=args.ckpt)

if __name__ == '__main__':
    main()
