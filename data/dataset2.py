# dataset2.py: with wav2lip postprocessing data 'all_wav2lip_coeffs_fns'
from glob import glob
import os
import numpy as np
import scipy.io as scio
import random
import torch.utils.data as data
# from .audio import load_wav, melspectrogram
from .audio import load_wav, melspectrogram
from PIL import Image
from skimage import io, img_as_float32, transform
import torchvision.transforms as tvtransforms
import torch
from math import ceil


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav


def parse_audio_length(num_frames, sr, fps):
    bit_per_frames = sr / fps
    audio_length = int(num_frames * bit_per_frames)
    return audio_length


class InferenceDataset(data.Dataset):  # audio and first frames
    def __init__(self, dataset_dir, subset, fps=30, sr=16000):
        # self.all_audio_fns = sorted(list(glob(f"{dataset_dir}/{subset}/audios/*.wav")))
        self.all_audio_fns = sorted(list(glob(f"{dataset_dir}/audios/*.wav")))
    
        # self.all_coeffs_fns = sorted(list(glob(f"{dataset_dir}/{subset}/recons/*speaker.mat")))
        # /media/cwy/sdb1/data/vico_challenge_2023/test/talking_head/inputs/recons/first_frames
        self.all_coeffs_fns = sorted(list(glob(f"{dataset_dir}/recons/first_frames/*speaker.mat")))

        # self.all_wav2lip_coeffs_fns = sorted(list(glob(f"{dataset_dir}/recons/wav2lip/*speaker.mat")))
        self.all_wav2lip_coeffs_fns = sorted(list(glob(f"{dataset_dir}/recons/wav2lip_gan/*speaker.mat")))

        # self.first_frames_fns = sorted(list(glob(f"{dataset_dir}/{subset}/frames/*")))
        # /media/cwy/sdb1/data/vico_challenge_2023/test/talking_head/inputs/first_frames
        self.first_frames_fns = sorted(list(glob(f"{dataset_dir}/first_frames/*.png")))

        # assert the frame name is the same with audio name and coeff name
        # assert len(self.all_coeffs_fns) == len(self.all_audio_fns), "num of files should match."
        # print(len(self.all_wav2lip_coeffs_fns))
        assert len(self.all_coeffs_fns) == len(self.all_audio_fns) == len(self.first_frames_fns) == len(self.all_wav2lip_coeffs_fns), "num of files should match."
        for i in range(len(self.all_audio_fns)):
            audio_name = self.all_audio_fns[i].split(os.path.sep)[-1].split('.')[0]
            coeff_name = self.all_coeffs_fns[i].split(os.path.sep)[-1].split('.')[0]
            wav2lip_coeff_name = self.all_wav2lip_coeffs_fns[i].split(os.path.sep)[-1].split('.')[0]
            frame_name = self.first_frames_fns[i].split(os.path.sep)[-1].split('.')[0]
            assert audio_name == coeff_name == frame_name == wav2lip_coeff_name
        print("InferenceDataset init done!")


        self.fps = fps
        self.sr = sr # TODO check 44100hz

    def __getitem__(self, idx):
        # generate batched audio_feats (bs=num_frames) for inference
        # load audio file
        wav = load_wav(path=self.all_audio_fns[idx], sr=self.sr)

        # get first frame: loading as SadTalker format
        # first_frame_fn = f'{self.first_frames_fns[idx]}/0000.jpg'
        first_frame_fn = f'{self.first_frames_fns[idx]}'
        first_frame = Image.open(first_frame_fn)
        first_frame = np.array(first_frame) # original shape: 256,256,3
        # print("original image shape:", first_frame.shape)
        first_frame = img_as_float32(first_frame)  # data range [0, 1]
        first_frame = transform.resize(first_frame, (256, 256, 3))
        first_frame = first_frame.transpose((2, 0, 1))  # 3, 256, 256

        # # load 3dmm_coeffs file
        # pred_coeffs: (1, 257)
        # pred_trans_params: (1, 5)
        mat_dict = scio.loadmat(self.all_coeffs_fns[idx])
        wav2lip_mat_dict = scio.loadmat(self.all_wav2lip_coeffs_fns[idx])

        wav2lip_exp = wav2lip_mat_dict['coeff'][:, 80:144]
        exp = mat_dict['coeff'][:, 80:144]
        angle = mat_dict['coeff'][:, 224:227]
        trans = mat_dict['coeff'][:, 254:257]
        pose = np.concatenate((angle, trans), axis=-1)
        crop = mat_dict['transform_params'][:, -3:] # TODO does this contain the head motion?

        # num_frames decided by wav length
        wav_ = load_wav(path=self.all_audio_fns[idx], sr=44100)
        num_frames = int(len(wav_) * self.fps / 44100) # 780

        # num_frames = math.floor(len(wav_)/44100)*self.fps
        wav_length = parse_audio_length(num_frames, sr=self.sr, fps=self.fps)
        wav = crop_pad_audio(wav, wav_length)
        orig_mel = melspectrogram(wav).T # hparams.sampling rate is 16000
        spec = orig_mel.copy() #(2867,40)
        # print("spec shape:",spec.shape)
        # print("num frames:",num_frames)

        # get mel sequence for coeff prediction
        mel_step_size = 5  # mel_window_size: 16 mel_frames
        fft_hop_length = 200 # TODO check this
        indiv_mels = []
        for i in range(0, num_frames):
            mel_fps = self.sr / fft_hop_length  # 16000/200=80  --> 22050/200
            start_frame_num = i - 1
            # start_frame_num = i
            mel_start_idx = int(mel_fps * start_frame_num / float(self.fps))
            mel_end_idx = mel_start_idx + mel_step_size
            # print(mel_start_idx,mel_end_idx)
            indexes = list(range(mel_start_idx, mel_end_idx))
            indexes = [min(max(item, 0), orig_mel.shape[0] - 1) for item in indexes]
            # print(indexes)
            m = spec[indexes, :]
            # print(m.shape)
            indiv_mels.append(m.T)
        indiv_mels = np.asarray(indiv_mels)  # num_frames 80 16 --> TODO num_frame, 40, 5
        # print('indiv_mels shape:',indiv_mels.shape)

        sample = {
            'first_frame': first_frame,  # 3, 256, 256
            'exp': exp,               # 1, 64 (init)
            'wav2lip_exp':wav2lip_exp,# num_frames, 64
            'pose': pose,             # 1, 6 (init)
            'crop': crop,             # 1, 3  for PIRenderer
            'indiv_mels': indiv_mels  # num_frames, 80, 16
        }
        return sample

    def __len__(self):
        return len(self.all_audio_fns)


class TrainingDataset_pre(data.Dataset):
    def __init__(self, dataset_dir, subset='train'):
        # self.all_batch_fns = sorted(list(glob(f"{dataset_dir}/{subset}/batches_len30/**/*.pt")))
        self.all_batch_fns = sorted(list(glob(f"{dataset_dir}/{subset}/batches_offset0/**/*.pt")))

    def __getitem__(self, idx):
        return torch.load(self.all_batch_fns[idx])

    def __len__(self):
        return len(self.all_batch_fns)

if __name__ == '__main__':
    # wav_path = '/media/cwy/sdb1/data/vico_challenge_2023/test/talking_head/inputs/audios/arj7h6n8k7le.wav'
    # wav_ = load_wav(path=wav_path, sr=44100)
    # wav = load_wav(path=wav_path, sr=22050)
    # print(len(wav))
    # print(len(wav_)/44100)
    # mel = melspectrogram(wav_)
    # print(mel.shape)

    dataset = InferenceDataset('/media/cwy/sdb1/data/vico_challenge_2023/test/talking_head/inputs',None)
    for video_idx in range(len(dataset)):
        audio_fn = dataset.all_audio_fns[video_idx]
        video_name = audio_fn.split('/')[-1].split('.')[0]
        print("video name:", video_name)
        data = dataset.__getitem__(video_idx)  # numpy
        

        first_frame = torch.from_numpy(data['first_frame']).cuda().float()  # (3, 256, 256)
        exp_init = torch.from_numpy(data['exp'])[:1, :].cuda().float()  # (1, 64)
        pose_init = torch.from_numpy(data['pose'])[:1, :].cuda().float()  # (1, 6)
        crop_init = torch.from_numpy(data['crop'])[:1, :].cuda().float()  # (1, 3)
        indiv_mels = torch.from_numpy(data['indiv_mels']).cuda().float()  # (num_frames, 80, 16)

        pose = torch.from_numpy(data['pose']).cuda().float()  # (num_frames, 6)  -> for now it's (1,6)
        crop = torch.from_numpy(data['crop']).cuda().float()  # (num_frames, 3)  -> for now it's (1,3)
        