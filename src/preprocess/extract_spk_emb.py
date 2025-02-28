import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from src.preprocess.audio import preprocess_wav, wav_to_mel_spectrogram
from glob import glob
from tqdm import tqdm
from src.models.speaker import SpeakerEncoder


def get_emb(filepath, model):
    audio = preprocess_wav(filepath)
    frames = wav_to_mel_spectrogram(audio)
    frames = torch.from_numpy(frames).cuda()
    with torch.no_grad():
        embed = model.forward(frames.unsqueeze(0))
    emb_filepath = filepath.replace('/audio-16k/', '/spk_emb_ge2e/').replace('.wav', '.pt')
    os.makedirs(os.path.dirname(emb_filepath), exist_ok=True)
    torch.save(embed, emb_filepath)


def get_filelist(dirpath):
    filelist = glob(os.path.join(dirpath, 'audio-16k', '*', '*', '*.wav'))
    return filelist


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--model_weight', type=str, default='')
    args = parser.parse_args()
    filelist = get_filelist(args.root)
    model = SpeakerEncoder('cuda', torch.device('cpu'))
    ckpt = torch.load(args.model_weight)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    for file in tqdm(filelist):
        get_emb(file, model)