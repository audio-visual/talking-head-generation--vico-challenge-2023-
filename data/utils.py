import os
from tqdm import tqdm
import torch
import numpy as np
import random
import scipy.io as scio
import audio


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav


def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames


def get_audio_data(audio_path):
    # generate batched audio_feats (bs=num_frames) for inference
    syncnet_mel_step_size = 16
    fps = 25

    wav = audio.load_wav(audio_path, 16000)
    wav_length, num_frames = parse_audio_length(len(wav), sr=16000, fps=25)
    wav = crop_pad_audio(wav, wav_length)
    orig_mel = audio.melspectrogram(wav).T
    spec = orig_mel.copy()         # nframes 80 ? n_mels
    indiv_mels = []

    for i in tqdm(range(num_frames), 'frame:'):
        start_frame_num = i-2
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
        m = spec[seq, :]
        indiv_mels.append(m.T)
    indiv_mels = np.asarray(indiv_mels)         # T 80 16
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0)  # bs T 1 80 16

    return indiv_mels
