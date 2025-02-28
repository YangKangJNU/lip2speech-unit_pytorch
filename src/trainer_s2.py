import os
import random
import torch
from torch.utils.data import Dataset
import torchvision
from src.utils import STFT, load_wav_to_torch

class MelCodeDataset(Dataset):
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
        self.build_unit_dict()
        self.stft = STFT(filter_length=self.filter_length, hop_length=self.hop_length, win_length=self.win_length, sampling_rate=self.sampling_rate, mel_fmin=self.mel_fmin, mel_fmax=self.mel_fmax)
        self.use_blur = cfg.use_blur
        self.use_noise = cfg.use_noise
        self.blur = torchvision.transforms.GaussianBlur(kernel_size=cfg.blur_kernel_size, sigma=(cfg.blur_sigma_min, cfg.blur_sigma_max))
        self.noise_factor = cfg.noise_factor
        self.max_window_size = 20

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
        audio_path = crop_path.replace('/crop/', '/audio/').replace('.mp4', '.wav')
        audio, mel = self.get_mel(audio_path)
        audio = torch.FloatTensor(audio)
        spk_path = crop_path.replace('/crop/', '/spk_emb_ge2e/').replace('.mp4', '.pt')
        spk_emb = torch.load(spk_path, map_location=torch.device('cpu'))

        units = torch.tensor(self.unit_dict[os.path.join(*crop_path.split("/")[-3:]).replace('.mp4', '.wav').replace('crop', 'audio')], dtype=torch.int32) + 1
        if self.mode == 'train':
            if units.size(0) > self.max_window_size:
                st_fr = random.randint(0, units.size(0) - self.max_window_size)
                units = units[st_fr:st_fr+self.max_window_size]
                mel = mel[:, st_fr*2:(st_fr+self.max_window_size)*2]
                audio = audio[st_fr*320:(st_fr+self.max_window_size)*320] 
        origin_mel = mel.clone()
        if self.use_blur:     
            mel = self.blur(mel.unsqueeze(0)).squeeze(0)
        if self.use_noise:
            mel = mel + torch.randn_like(mel) * self.noise_factor
        diff = mel.size(1) - len(units)*2
        if diff < 0:
            padding_mel = torch.zeros(mel.size(0), -diff, dtype=mel.dtype)
            mel = torch.cat((mel, padding_mel), dim=-1)
            origin_mel = torch.cat((origin_mel, padding_mel), dim=-1)
        elif diff > 0:
            mel = mel[:, :-diff]
            origin_mel = origin_mel[:, :-diff]

        diff = audio.size(0) - len(units)*320
        if diff < 0:
            padding_audio = torch.zeros(-diff, dtype=audio.dtype)
            audio = torch.cat((audio, padding_audio), dim=0)
        elif diff > 0:
            audio = audio[:-diff]              
        return (units, mel, origin_mel, audio, spk_emb.squeeze(0), f_name)
    
def collate_fn(batch):
    _, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[0].size(0) for x in batch]),
        dim=0, descending=True)
    
    max_units_len = max(len(x[0]) for x in batch)
    max_mel_len = max(x[1].size(1) for x in batch)
    max_omel_len = max(x[2].size(1) for x in batch)
    max_audio_len = max(x[3].size(0) for x in batch)

    units_lengths = torch.IntTensor(len(batch))
    mel_lengths = torch.IntTensor(len(batch))
    omel_lengths = torch.IntTensor(len(batch))
    audio_lengths = torch.IntTensor(len(batch))    

    units_padded = torch.zeros(len(batch), max_units_len, dtype=torch.int64)
    mel_padded = torch.zeros(len(batch), 80, max_mel_len, dtype=torch.float32)
    omel_padded = torch.zeros(len(batch), 80, max_omel_len, dtype=torch.float32)
    audio_padded = torch.zeros(len(batch), max_audio_len, dtype=torch.float32)

    units_padded.zero_()
    mel_padded.zero_()
    omel_padded.zero_()
    audio_padded.zero_()
  
    spks = [] 
    file_name = []


    for i in range(len(ids_sorted_decreasing)):
        row = batch[ids_sorted_decreasing[i]]    

        units = row[0]
        units_padded[i, :units.size(0)] = units
        units_lengths[i] = units.size(0)
        
        mel = row[1]
        mel_padded[i, :, :mel.size(1)] = mel
        mel_lengths[i] = mel.size(1)

        omel = row[1]
        omel_padded[i, :, :omel.size(1)] = omel
        omel_lengths[i] = omel.size(1)

        audio = row[3]
        audio_padded[i, :audio.size(0)] = audio
        audio_lengths[i] = audio.size(0)

        spks.append(row[4])
        file_name.append(row[5])

    spks = torch.stack(spks)

    return dict(
        units = units_padded,
        units_lengths = units_lengths,
        mel = mel_padded,
        omel = omel_padded,
        mel_lengths = mel_lengths,
        audio = audio_padded,
        audio_lengths = audio_lengths,
        spk_emb = spks,
        file_name = file_name
    )

