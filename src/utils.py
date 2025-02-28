import glob
import os
import random
import cv2
import numpy as np
import torch
from scipy.io.wavfile import read
from src.stft import TacotronSTFT
import matplotlib.pylab as plt
from librosa.filters import mel as librosa_mel_fn

mel_basis = {}
hann_window = {}

def default(v, d):
    return v if exists(v) else d

def exists(v):
    return v is not None

def lens_to_mask(t, length = None) -> bool:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]

def load_video(path):
    for i in range(3):
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                else:
                    break
            frames = np.stack(frames)
            return frames
        except Exception:
            print(f"failed loading {path} ({i} / 3)")
            if i == 2:
                raise ValueError(f"Unable to load {path}")

class Compose(object):
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw)) / 2.)
        delta_h = int(round((h - th)) / 2.)
        frames = frames[:, delta_h:delta_h + th, delta_w:delta_w + tw]
        return frames

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w - tw)
        delta_h = random.randint(0, h - th)
        frames = frames[:, delta_h:delta_h + th, delta_w:delta_w + tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class HorizontalFlip(object):
    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames

class TimeMask(object):
    """time mask
    """
    def __init__(self, max_mask_T=0.4, hop_T=1., fps=25, replace_with_zero=True, inplace=False):
        self.max_mask_frame = round(max_mask_T * fps)
        self.hop_frame = round(hop_T * fps)

        self.replace_with_zero = replace_with_zero
        self.inplace = inplace

    def __call__(self, x):
        if self.inplace:
            cloned = x
        else:
            cloned = x.copy()

        len_raw = cloned.shape[0]

        for i in range(len_raw//self.hop_frame):
            mask_len = random.randint(0, self.max_mask_frame)
            mask_start = random.randint(0, self.hop_frame - mask_len)
            if self.replace_with_zero:
                cloned[i*self.hop_frame+mask_start : i*self.hop_frame+mask_start+mask_len] = 0.
            else:
                cloned[i*self.hop_frame+mask_start : i*self.hop_frame+mask_start+mask_len] = cloned.mean()

        return cloned

import math
class RandomErase(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), replace_with_zero=True):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.replace_with_zero = replace_with_zero

    def get_params(self, frames, scale, ratio):
        t, h, w = frames.shape
        area = h * w

        log_ratio = np.log(np.array(ratio))
        while True:
            erase_area = area * random.uniform(scale[0], scale[1])
            aspect_ratio = np.exp(random.uniform(log_ratio[0], log_ratio[1]))

            erase_h = int(round(math.sqrt(erase_area * aspect_ratio)))
            erase_w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if (erase_h < h and erase_w < w):
                i = random.randint(0, h - erase_h)
                j = random.randint(0, w - erase_w)
                return i, j, h, w

    def __call__(self, frames):
        if random.random() < self.p:
            i, j, h, w = self.get_params(frames, scale=self.scale, ratio=self.ratio)
            if self.replace_with_zero:
                frames[:, i:i+h, j:j+w] = 0.
            else:
                frames[:, i:i+h, j:j+w] = frames.mean()

        return frames
    
class STFT():
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, filter_length, hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)

    def get_mel(self, audio):
        audio = audio.unsqueeze(0)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0)
        return melspec
    
def normalise_mel(melspec, min_val=math.log(1e-5)): 
    melspec = ((melspec - min_val) / (-min_val / 2)) - 1    #log(1e-5)~2 --> -1~1
    return melspec

def denormalise_mel(melspec, min_val=math.log(1e-5)): 
    melspec = ((melspec + 1) * (-min_val / 2)) + min_val
    return melspec

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

def to_numpy(t):
    return t.detach().cpu().numpy()

def plot_spectrogram(spectrogram):
    spectrogram = to_numpy(spectrogram)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(spectrogram.T, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    plt.close()
    return fig

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def plot_spectrogram_vocoder(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig