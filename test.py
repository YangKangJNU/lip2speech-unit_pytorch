



import argparse
import os

import torch
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import hydra
from omegaconf import DictConfig
from src.trainer_s1 import (
    VADataset,
    Trainer,
    collate_fn
)
from torch.utils.data import DataLoader
from src.models.model import L2SUnit
from src.models.vocoder import MelCodeGenerator
from src.utils import load_checkpoint
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--vocoder_ckpt', default='')
    args = parser.parse_args()
    return args

@hydra.main(version_base=None, config_path="configs/v1", config_name="default.yaml")
def main(cfg: DictConfig):
    args = parse_args()
    test_dataset = VADataset('test', cfg.data)
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn, shuffle=False, num_workers=16, pin_memory=True)
    device = torch.device('cuda')
    model = L2SUnit(cfg.encoder).to(device)
    generator = MelCodeGenerator(cfg.vocoder).to(device)
    state_dict_g = load_checkpoint(args.vocoder_ckpt, device)
    generator.load_state_dict(state_dict_g['generator'])
    trainer = Trainer(
        model=model,
        vocoder=generator,
    )
    trainer.test(test_dataloader, args.output_dir, args.ckpt, device)

if __name__ == '__main__':
    main()