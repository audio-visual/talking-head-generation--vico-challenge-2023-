import os
import argparse
from tqdm import tqdm
from glob import glob
import torch
import numpy as np
import scipy.io as scio
from data.audio import load_wav, melspectrogram
from utils_parallel import get_config
from math import ceil


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/media/cwy/sdb1/data/vico_challenge_2023')
    parser.add_argument('--subset', type=str, default='train')
    parser.add_argument('--config', type=str, default='configs/train_transformer.yaml', help='Path to the config file.')
    parser.add_argument("--len_clip", type=int, default=90)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir  # '../vico_dataset_'
    subset = args.subset  # 'train'

    all_audio_fns = sorted(list(glob(f"{dataset_dir}/{subset}/audios/*.wav"))) #430
    all_coeffs_fns = sorted(list(glob(f"{dataset_dir}/{subset}/recons/*speaker.mat"))) # 实际上没有wav2lip_recons
    # all_lip_coeffs_fns = sorted(list(glob(f"{dataset_dir}/{subset}/wav2lip_recons/*speaker.mat")))
    # all_frames_dirs = sorted(list(glob(f"{dataset_dir}/{subset}/frames/*")))

    # assert len(all_coeffs_fns) == len(all_lip_coeffs_fns), "num of files should match."

    # generate batched audio_feats (bs=num_frames) for inference
    config = get_config(args.config)
    fps = config['fps'] #30
    sr = config['sr'] # 16000
    len_clip = config['model']['embed_len'] # 90
   

    data_sample_count = 0

    for idx in tqdm(range(len(all_audio_fns))):

        # load audio file
        wav = load_wav(path=all_audio_fns[idx], sr=config['sr'])
        # load 3dmm_coeffs file
        mat_dict = scio.loadmat(all_coeffs_fns[idx])
        exp = mat_dict['coeff'][:, 80:144] # (592,64)
        angle = mat_dict['coeff'][:, 224:227]
        trans = mat_dict['coeff'][:, 254:257]
        pose = np.concatenate((angle, trans), axis=1)
        crop = mat_dict['transform_params'][:, -3:]

        # lip_mat_dict = scio.loadmat(all_lip_coeffs_fns[idx])
        lip_mat_dict = mat_dict.copy()
        lip_exp = lip_mat_dict['coeff'][:, 80:144]


        num_frames = len(lip_exp)
       
        # the number of audio sampling points for a frame
        bit_per_frames = sr / fps
        
        wav_length = int(num_frames * bit_per_frames)
        wav = crop_pad_audio(wav, wav_length)
        orig_mel = melspectrogram(wav).T
        spec = orig_mel.copy() # (1579,40)

        # data samples from random fetching
        start_frame_idx = 0
        # train len_clip length of data at once
        """
        training:
          inputs: first frame's coeff + a period of time's audio mel features (time=5 mel frames) 
          targets: a period of time's coeff
          training length: 90
        """
        while start_frame_idx < num_frames-len_clip-10:
            start_frame_idx += 3

            # clip coeffs
            # used as inputs
            exp_init = exp[start_frame_idx, :]
            pose_init = pose[start_frame_idx, :]
            crop_init = crop[start_frame_idx, :]

            # len_clip=90
            # used as prediction targets
            exp_clip = exp[start_frame_idx:(start_frame_idx + len_clip), :]
            pose_clip = pose[start_frame_idx:(start_frame_idx + len_clip), :]
            crop_clip = crop[start_frame_idx:(start_frame_idx + len_clip), :]

            lip_exp_clip = lip_exp[start_frame_idx:(start_frame_idx + len_clip), :]
            try:
                assert lip_exp_clip.shape[0] == len_clip
            except:
                print(lip_exp_clip.shape[0])

            # get mel sequence for coeff prediction
            mel_step_size = 5 # for every 5 mel frames, need to predict 1 frame coeffs
            fft_hop_length = 200
            indiv_mels = []

            for i in range(start_frame_idx, start_frame_idx + len_clip):
                start_frame_num = i 
                mel_fps = sr / fft_hop_length  # 16000/200=80  
                # start_frame_num / float(fps) : the current seconds
                mel_start_idx = int(mel_fps * start_frame_num / float(fps))
               
                mel_end_idx = mel_start_idx + mel_step_size
                indexes = list(range(mel_start_idx, mel_end_idx))
                indexes = [min(max(item, 0), orig_mel.shape[0] - 1) for item in indexes]
                m = spec[indexes, :] # (5,40)
                indiv_mels.append(m.T) # (40,5)
            indiv_mels = np.asarray(indiv_mels)  # (90,40,5)

            data_sample = {
                'exp_init': exp_init,  # 64
                'pose_init': pose_init,  # 6
                'crop_init': crop_init, # 3
                'exp': exp_clip,  # T, 64 (T: syncnet_T=5)
                'pose': pose_clip,  # T, 6
                'crop': crop_clip,  # 90, 3  for PIRenderer
                'lip_exp': lip_exp_clip,  # T, 64 (T: syncnet_T=5)
                'indiv_mels': indiv_mels  # 90,40,5
            }

            video_name = all_audio_fns[idx].split('/')[-1].split('.')[-2]  # get video_name
            os.makedirs(os.path.join(dataset_dir, subset, 'batches_offset0', video_name), exist_ok=True)
            torch.save(data_sample, f'{dataset_dir}/{subset}/batches_offset0/{video_name}/{str(start_frame_idx).zfill(4)}.pt')

            data_sample_count += 1

    print(data_sample_count)
