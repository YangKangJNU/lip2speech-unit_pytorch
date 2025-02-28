import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.utils import load_video, Compose, Normalize, RandomCrop, HorizontalFlip, CenterCrop, STFT, normalise_mel, denormalise_mel, load_wav_to_torch, TimeMask, RandomErase
from loguru import logger
from accelerate.utils import DistributedDataParallelKwargs
from adam_atan2_pytorch.adopt import Adopt
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from src.utils import default, exists, plot_spectrogram
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import soundfile as sf

class VADataset(Dataset):
    def __init__(self, mode, cfg=None):
        self.root = cfg.root
        self.mode = mode
        self.filter_length = cfg.filter_length
        self.hop_length = cfg.hop_length
        self.win_length = cfg.win_length
        self.sampling_rate = cfg.sampling_rate
        self.mel_fmin = cfg.mel_fmin
        self.mel_fmax = cfg.mel_fmax
        self.file_paths, self.file_names = self.build_file_list(self.root, self.mode)
        if mode == 'train':
            self.video_transform = Compose([
                    Normalize(0.0, 255.0),
                    RandomCrop((88, 88)),
                    HorizontalFlip(0.5),
                    Normalize(0.421, 0.165),
                    TimeMask(),
                    RandomErase(0.5)])
        else:
            self.video_transform = Compose([
                    Normalize(0.0, 255.0),
                    CenterCrop((88, 88)),
                    Normalize(0.421, 0.165)])
        self.build_unit_dict()
        self.stft = STFT(filter_length=self.filter_length, hop_length=self.hop_length, win_length=self.win_length, sampling_rate=self.sampling_rate, mel_fmin=self.mel_fmin, mel_fmax=self.mel_fmax)
        self.max_window_size = cfg.max_window_size
        self.min_window_size = cfg.min_window_size


    def build_file_list(self, root, mode):
        file_list, paths = [], []
        assert mode in ['train', 'val', 'test']
        with open(f"{root}/{mode}.csv", "r") as f:
            train_data = f.readlines()
        for i in range(len(train_data)):
            file = train_data[i].strip().replace('_', '/', 1)
            file_list.append(file)
            paths.append(f"{root}/crop/{file}.mp4")
        return paths, file_list

    def build_unit_dict(self):
        base_fname_batch, quantized_units_batch = [], []
        units_file = ''
        with open(units_file) as f:
            for line in f:
                base_fname, quantized_units_str = line.rstrip().split("|")
                quantized_units = [int(q) for q in quantized_units_str.split(" ")]
                base_fname_batch.append(base_fname)
                quantized_units_batch.append(quantized_units)
        self.unit_dict = dict(zip(base_fname_batch,quantized_units_batch))

    def __len__(self):
        return len(self.file_names)

    def get_mel(self, filename):
        audio, _ = load_wav_to_torch(filename)
        audio = audio / 1.1 / audio.abs().max()
        melspectrogram = self.stft.get_mel(audio)
        return audio, melspectrogram
    
    def __getitem__(self, index):
        crop_path = self.file_paths[index]
        f_name = self.file_names[index]
        video = load_video(crop_path)
        video = np.array(video, dtype=np.float32)
        video = self.video_transform(video)
        video = torch.tensor(video).unsqueeze(0)
        audio_path = crop_path.replace('/crop/', '/audio/').replace('.mp4', '.wav')
        audio, mel = self.get_mel(audio_path)
        audio = torch.FloatTensor(audio)
        # mel = normalise_mel(mel)
        spk_path = crop_path.replace('/crop/', '/spk_emb_ge2e/').replace('.mp4', '.pt')
        spk_emb = torch.load(spk_path, map_location=torch.device('cpu'))
        units = torch.tensor(self.unit_dict[os.path.join(*crop_path.split("/")[-3:]).replace('.mp4', '.wav').replace('crop', 'audio')], dtype=torch.int32) + 1
        if self.mode == 'train':
            if video.size(1) > self.max_window_size:
                start_fr = random.randint(0, video.size(1) - self.max_window_size)
                video = video[:, start_fr:start_fr+self.max_window_size, :, :]
                units = units[start_fr*2:(start_fr+self.max_window_size)*2]
                mel = mel[:, start_fr*4:(start_fr+self.max_window_size)*4]
                audio = audio[start_fr*640:(start_fr+self.max_window_size)*640]

            if video.size(1) < self.min_window_size:
                video_padded = torch.zeros(1, self.min_window_size-video.size(1), 88, 88)
                video_padded.zero_()
                video = torch.cat((video, video_padded), dim=1)

        diff = units.size(0) - video.size(1)*2
        if diff < 0:
            padding_units = torch.zeros(-diff, dtype=units.dtype)
            units = torch.cat((units, padding_units), dim=-1)
        elif diff > 0:
            units = units[:-diff]        

        diff = mel.size(1) - video.size(1)*4
        if diff < 0:
            padding_mel = torch.zeros(mel.size(0), -diff, dtype=mel.dtype)
            mel = torch.cat((mel, padding_mel), dim=-1)
        elif diff > 0:
            mel = mel[:, :-diff]

        diff = audio.size(0) - video.size(1)*640
        if diff < 0:
            padding_audio = torch.zeros(-diff, dtype=audio.dtype)
            audio = torch.cat((audio, padding_audio), dim=0)
        elif diff > 0:
            audio = audio[:-diff]              
        
        return (video, units, mel, audio, spk_emb.squeeze(0), f_name)



def collate_fn(batch):
    _, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[0].size(1) for x in batch]),
        dim=0, descending=True)
    
    max_video_len = max([x[0].size(1) for x in batch])
    max_units_len = max(len(x[1]) for x in batch)
    max_mel_len = max(x[2].size(1) for x in batch)
    max_audio_len = max(x[3].size(0) for x in batch)

    video_lengths = torch.IntTensor(len(batch))
    units_lengths = torch.IntTensor(len(batch))
    mel_lengths = torch.IntTensor(len(batch))
    audio_lengths = torch.IntTensor(len(batch))    

    video_padded = torch.zeros(len(batch), 1, max_video_len, 88, 88, dtype=torch.float32)
    units_padded = torch.zeros(len(batch), max_units_len, dtype=torch.int64)
    mel_padded = torch.zeros(len(batch), 80, max_mel_len, dtype=torch.float32)
    audio_padded = torch.zeros(len(batch), max_audio_len, dtype=torch.float32)

    video_padded.zero_()
    units_padded.zero_()
    mel_padded.zero_()
    audio_padded.zero_()

    spks = []   
    file_name = []


    for i in range(len(ids_sorted_decreasing)):
        row = batch[ids_sorted_decreasing[i]]

        video = row[0]
        video_padded[i, :, :video.size(1), :, :] = video
        video_lengths[i] = video.size(1)      

        units = row[1]
        units_padded[i, :units.size(0)] = units
        units_lengths[i] = units.size(0)
        

        mel = row[2]
        mel_padded[i, :, :mel.size(1)] = mel
        mel_lengths[i] = mel.size(1)

        audio = row[3]
        audio_padded[i, :audio.size(0)] = audio
        audio_lengths[i] = audio.size(0)

        spks.append(row[4])
        file_name.append(row[5])

        

    spks = torch.stack(spks)


    return dict(
        video = video_padded,
        video_lengths = video_lengths,
        units = units_padded,
        units_lengths = units_lengths,
        mel = mel_padded,
        mel_lengths = mel_lengths,
        audio = audio_padded,
        audio_lengths = audio_lengths,
        spk_emb = spks,
        file_name = file_name
    )

class Trainer:
    def __init__(
            self,
            model,
            vocoder=None,
            lr = 1e-4,
            checkpoint_path = None,
            grad_accumulation_steps = 1,
            log_file = 'logs.txt',
            tensorboard_log_dir = 'logs/test1',
            accelerate_kwargs: dict = dict(),
            num_warmup_steps = None,
            sample_steps = 50,
            max_grad_norm = 1.0,
    ):
        logger.add(log_file)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
        self.accelerator = Accelerator(
            log_with = "all",
            kwargs_handlers = [ddp_kwargs],
            gradient_accumulation_steps = grad_accumulation_steps,
            mixed_precision='no',
            **accelerate_kwargs
        )        
        self.model = model
        self.vocoder = vocoder
        self.lr = lr
        self.ema_model = None
        self.optimizer = None
        self.num_warmup_steps = num_warmup_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_path = default(checkpoint_path, 'model.pth') 
        self.sample_steps = sample_steps
        self.tensorboard_log_dir = tensorboard_log_dir

        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.scheduler = None

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    def save_checkpoint(self, step, ckpt_name):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict = self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict = self.accelerator.unwrap_model(self.optimizer).state_dict(),
                scheduler_state_dict = self.scheduler.state_dict(),
                step = step
            )

            self.accelerator.save(checkpoint, ckpt_name)   

    def load_checkpoint(self, ckpt_path):
        if not exists(ckpt_path) or not os.path.exists(ckpt_path):
            logger.info('not found ckpt')
            return 0

        checkpoint = torch.load(ckpt_path)
        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer:
            self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logger.info('not load optimizer')

        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            logger.info('not load scheduler')
        return checkpoint['step']
    
    def train(self, cfg, train_dataset, val_dataset=None, ckpt_path=None):
        self.optimizer = Adopt(self.model.parameters(), lr = self.lr)
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=cfg.num_workers, pin_memory=False)
        val_dataloder = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True, num_workers=1, pin_memory=False)
        total_steps = len(train_dataloader) * cfg.epochs
        decay_steps = total_steps - self.num_warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.num_warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_steps)
        self.scheduler = SequentialLR(self.optimizer, 
                                      schedulers=[warmup_scheduler, decay_scheduler],
                                      milestones=[self.num_warmup_steps])
        self.model, train_dataloader, val_dataloder, self.scheduler = self.accelerator.prepare(self.model, train_dataloader, val_dataloder, self.scheduler)
        if self.vocoder is not None:
            self.vocoder = self.accelerator.prepare(self.vocoder)
        start_step = self.load_checkpoint(ckpt_path)
        global_step = start_step
        start_epoch = start_step // len(train_dataloader) - 1
        ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
        mel_loss_fn = torch.nn.L1Loss()
        for epoch in range(start_epoch, cfg.epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch in tqdm(train_dataloader):
                with self.accelerator.accumulate(self.model):            
                    video = batch['video']
                    video_len = batch['video_lengths']
                    spk_emb = batch['spk_emb']
                    hubert_unit = batch['units']
                    mel = batch['mel']
                    mel_hat, logits = self.model(video, video_len, spk_emb)
                    ce_loss = ce_loss_fn(logits.permute(0, 2, 1), hubert_unit-1)
                    acc = ((logits.argmax(dim=-1) == hubert_unit-1).sum() / ((hubert_unit-1) != -1).sum()).item()
                    mel_loss = mel_loss_fn(mel_hat, mel.permute(0, 2, 1))
                    total_loss = cfg.w_ce*ce_loss + cfg.w_mel*mel_loss
                    self.accelerator.backward(total_loss)
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                if global_step % cfg.log_step == 0:
                    if self.accelerator.is_local_main_process:        
                        logger.info(f"step {global_step+1}: ce_loss = {ce_loss.item():.4f}, mel_loss = {mel_loss.item():.4f}, acc = {np.mean(acc):.4f}")
                        lr = self.scheduler.get_last_lr()[0]
                        self.writer.add_scalar('train/ce_loss', ce_loss.item(), global_step)
                        self.writer.add_scalar('train/mel_loss', mel_loss.item(), global_step)
                        self.writer.add_scalar('train/lr', lr, global_step)
                        self.writer.add_scalar('train/acc', acc, global_step)
                        self.writer.add_figure('trian/mel_gt', plot_spectrogram(mel[0, :, :(video_len[0]*4).item()].permute(1, 0)), global_step)
                        self.writer.add_figure('trian/mel_hat', plot_spectrogram(mel_hat[0, :(video_len[0]*4).item(), :]), global_step)

                if global_step % cfg.eval_steps == 0:
                    eval_acc, eval_melerr = self.evaluate(val_dataloder, global_step, cfg.eval_sample_steps)
                    if self.accelerator.is_local_main_process: 
                        logger.info(f"step {global_step+1}: eval_acc = {eval_acc:.3f}, eval_melerr = {eval_melerr:.3f}")

                if global_step % cfg.save_steps == 0:
                    self.save_checkpoint(global_step, os.path.join(self.tensorboard_log_dir, 'model_{}_{:.3f}_{:.3f}.pt'.format(global_step, eval_acc, eval_melerr)))

                global_step += 1
                epoch_loss += total_loss.item()
            epoch_loss /= len(train_dataloader)
            if self.accelerator.is_local_main_process:
                logger.info(f"epoch {epoch+1}/{cfg.epochs} - average loss = {epoch_loss:.4f}")
                self.writer.add_scalar('epoch average loss', epoch_loss, epoch)

        self.writer.close()    

    def evaluate(self, val_dataloader, global_step, eval_sample_steps=50):
        mel_loss_fn = torch.nn.L1Loss()
        accs = []
        mel_losses = []
        self.model.eval()
        eval_step = 0
        for batch in tqdm(val_dataloader):
            with torch.no_grad():
                video = batch['video']
                video_len = batch['video_lengths']
                spk_emb = batch['spk_emb']
                hubert_unit = batch['units']
                mel = batch['mel']
                wav = batch['audio']
                mel_hat, logits = self.model(video, video_len, spk_emb)
                acc = ((logits.argmax(dim=-1) == hubert_unit-1).sum() / ((hubert_unit-1) != -1).sum()).item()
                mel_loss = mel_loss_fn(mel_hat, mel.permute(0, 2, 1))
                accs.append(acc)
                mel_losses.append(mel_loss.item())
                eval_step += 1
                if eval_step >= eval_sample_steps:
                    break

        if self.vocoder is not None:
            units = logits.argmax(dim=-1)
            wav_hat = self.vocoder(mel_hat, units)
        if self.accelerator.is_local_main_process:
            self.writer.add_scalar('eval/acc', np.mean(accs), global_step)
            self.writer.add_scalar('eval/mel_err', np.mean(mel_losses), global_step)
            for _ in range(mel_hat.size(0)):
                self.writer.add_figure(f'eval/mel_gt_{_}', plot_spectrogram(mel[_, :, :(video_len[_]*4).item()].permute(1, 0)), global_step)
                self.writer.add_figure(f'eval/mel_hat_{_}', plot_spectrogram(mel_hat[_, :(video_len[_]*4).item(), :]), global_step)
                if self.vocoder is not None:
                    self.writer.add_audio(f'eval/wav_gt_{_}', wav[_], global_step, sample_rate=16000)
                    self.writer.add_audio(f'eval/wav_hat_{_}', wav_hat[_], global_step, sample_rate=16000)
                

        return np.mean(accs), np.mean(mel_losses)
                
    def test(self, test_dataloader, output_dir, ckpt_path, device):
        _ = self.load_checkpoint(ckpt_path)
        self.model.eval()
        self.vocoder.eval()
        for batch in tqdm(test_dataloader):
            file_names = batch['file_name']
            exist_files = 0
            for file in file_names:
                if os.path.exists(os.path.join(output_dir, 'wav', file+'.wav')):
                    exist_files += 1
            if exist_files == len(file_names):
                continue   
            video = batch['video'].to(device)
            video_len = batch['video_lengths'].to(device)
            spk_emb = batch['spk_emb'].to(device)
            hubert_unit = batch['units'].to(device)
            mel = batch['mel'].to(device)
            mel_hat, logits = self.model(video, video_len, spk_emb)
            predict_units = logits.argmax(dim=-1)+1
            y_g_hat = self.vocoder(mel_hat.permute(0, 2 ,1), predict_units, spk_emb).squeeze(1)
            target_lens = video_len*640
            for b in range(y_g_hat.size(0)):
                m_name = file_names[b].split('/')[-3]
                v_name = file_names[b].split('/')[-2]
                file_name = file_names[b].split('/')[-1]+'.wav'

                if not os.path.exists(os.path.join(output_dir, 'wav', m_name, v_name)):
                    os.makedirs(os.path.join(output_dir, 'wav', m_name, v_name))

                sf.write(os.path.join(output_dir, 'wav', m_name, v_name, file_name), y_g_hat[b][:target_lens[b]].unsqueeze(1).cpu().detach().numpy(), 16000, subtype='PCM_16')
            