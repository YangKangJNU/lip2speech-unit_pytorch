

import argparse
import itertools
import os

from tqdm import tqdm
os.environ['HYDRA_FULL_ERROR'] = '1'

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from src.trainer_s2 import MelCodeDataset, collate_fn
from src.models.vocoder import MelCodeGenerator, MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss, generator_loss, feature_loss
from src.utils import scan_checkpoint, load_checkpoint, mel_spectrogram, save_checkpoint, plot_spectrogram_vocoder
import torch.nn.functional as F
import numpy as np

def train(a, cfg):
    device = torch.device('cuda')
    generator = MelCodeGenerator(cfg.vocoder).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    os.makedirs(a.output_dir, exist_ok=True)
    print("checkpoints directory : ", a.output_dir)

    if os.path.isdir(a.output_dir):
        cp_g = scan_checkpoint(a.output_dir, 'g_')
        cp_do = scan_checkpoint(a.output_dir, 'do_') 

    steps = 0

    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch'] 

    optim_g = torch.optim.AdamW(generator.parameters(), cfg.vocoder.lr, betas=[cfg.vocoder.adam_b1, cfg.vocoder.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), cfg.vocoder.lr,
                                betas=[cfg.vocoder.adam_b1, cfg.vocoder.adam_b2]) 
        
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=cfg.vocoder.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=cfg.vocoder.lr_decay, last_epoch=last_epoch)

    train_dataset = MelCodeDataset('train', cfg.data)
    val_dataset = MelCodeDataset('val', cfg.data)

    train_loader = DataLoader(train_dataset, num_workers=cfg.vocoder.num_workers, shuffle=True,
                              batch_size=cfg.vocoder.batch_size, pin_memory=True, drop_last=False, collate_fn=collate_fn)
    validation_loader = DataLoader(val_dataset, num_workers=cfg.vocoder.num_workers, shuffle=False, sampler=None,
                                batch_size=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    sw = SummaryWriter(os.path.join(a.output_dir))

    generator.train()
    mpd.train()
    msd.train()

    for epoch in range(max(0, last_epoch), a.epochs):
        for batch in tqdm(train_loader):
            code = torch.autograd.Variable(batch['units'].to(device, non_blocking=False))
            mel_noise = torch.autograd.Variable(batch['mel'].to(device, non_blocking=False))
            mel_origin = batch['omel']
            y = torch.autograd.Variable(batch['audio'].unsqueeze(1).to(device, non_blocking=False))
            spk = torch.autograd.Variable(batch['spk_emb'].to(device, non_blocking=False))
            f_name = batch['file_name']
            y_g_hat = generator(mel_noise, code, spk)

            assert y.shape == y_g_hat.shape, f"Mismatch in vocoder output shape - {y_g_hat.shape} != {y.shape}"

            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), cfg.vocoder.n_fft, cfg.vocoder.num_mels, cfg.vocoder.sampling_rate, cfg.vocoder.hop_size,
                                          cfg.vocoder.win_size, cfg.vocoder.fmin, cfg.vocoder.fmax_for_loss)
            y_mel = mel_spectrogram(y.squeeze(1), cfg.vocoder.n_fft, cfg.vocoder.num_mels, cfg.vocoder.sampling_rate, cfg.vocoder.hop_size,
                                          cfg.vocoder.win_size, cfg.vocoder.fmin, cfg.vocoder.fmax_for_loss)
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()

            if steps % cfg.vocoder.log_steps == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}'.format(steps, loss_gen_all, mel_error))
                
            if steps % cfg.vocoder.save_steps == 0 and steps != 0:
            # if steps % cfg.vocoder.save_steps == 0:
                checkpoint_path = "{}/g_{:08d}".format(a.output_dir, steps)
                save_checkpoint(checkpoint_path,
                                {'generator': generator.state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(a.output_dir, steps)
                save_checkpoint(checkpoint_path, {'mpd': mpd.state_dict(),
                                                    'msd': msd.state_dict(),
                                                    'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(),
                                                    'steps': steps, 'epoch': epoch})                    
            if steps % cfg.vocoder.summary_steps == 0:
                sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                sw.add_scalar("training/mel_spec_error", mel_error, steps)

            if steps % cfg.vocoder.val_steps == 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = []
                eval_sample_steps = 0
                with torch.no_grad():
                    for batch in tqdm(validation_loader):
                        code = batch['units'].to(device, non_blocking=False)
                        mel_noise = batch['mel'].to(device, non_blocking=False)
                        y = batch['audio'].unsqueeze(1).to(device, non_blocking=False)
                        spk = batch['spk_emb'].to(device, non_blocking=False)
                        y_g_hat = generator(mel_noise, code, spk)
                        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), cfg.vocoder.n_fft, cfg.vocoder.num_mels, cfg.vocoder.sampling_rate,
                                                        cfg.vocoder.hop_size, cfg.vocoder.win_size, cfg.vocoder.fmin, cfg.vocoder.fmax_for_loss)
                        y_mel = mel_spectrogram(y.squeeze(1), cfg.vocoder.n_fft, cfg.vocoder.num_mels, cfg.vocoder.sampling_rate,
                                                        cfg.vocoder.hop_size, cfg.vocoder.win_size, cfg.vocoder.fmin, cfg.vocoder.fmax_for_loss)
                        val_err_tot.append(F.l1_loss(y_mel, y_g_hat_mel).item())

                        if eval_sample_steps == 0:
                            if steps == 0:
                                for _ in range(code.shape[0]):
                                    sw.add_audio(f'gt/y_{_}', y[_], steps, cfg.vocoder.sampling_rate)
                                    sw.add_figure(f'gt/y_spec_{_}', plot_spectrogram_vocoder(y_mel[_].cpu()), steps)
                            for _ in range(code.shape[0]):
                                sw.add_audio(f'generated/y_hat_{_}', y_g_hat[_], steps, cfg.vocoder.sampling_rate)
                                sw.add_figure(f'generated/y_hat_spec_{_}',
                                            plot_spectrogram_vocoder(y_g_hat_mel[_].squeeze(0).cpu().numpy()), steps)
                        eval_sample_steps += 1
                        if eval_sample_steps >= cfg.vocoder.val_sample_steps:
                            break

                    val_err = np.mean(val_err_tot)
                    sw.add_scalar("validation/mel_spec_error", val_err, steps)
                generator.train()                
            steps += 1
        scheduler_g.step()
        scheduler_d.step()        



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2000)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--ckpt', default='')
    parser.add_argument('--lr', default=1e-4)
    args = parser.parse_args()
    return args



@hydra.main(version_base=None, config_path="configs/v1", config_name="default.yaml")
def main(cfg: DictConfig):
    args = parse_args()
    torch.manual_seed(cfg.vocoder.seed)
    train(args, cfg)

if __name__ == '__main__':
    main()
