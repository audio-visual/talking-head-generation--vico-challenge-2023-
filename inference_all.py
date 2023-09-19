import os
import argparse
import shutil
import numpy as np
from skimage import img_as_ubyte
import imageio
import cv2
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from logger import create_logger
from data.dataset import InferenceDataset
from data.generate_facerender_batch import get_facerender_data
from models.disentangle_decoder.disentangle_decoder import DisentangleDecoder
from models.transformer.transformer import Transformer3DMM
from models.lstm_decoder.LSTMStepwiseFusion import LSTMStepwiseFusion
from models.facerender.animate import AnimateFromCoeff
from utils_parallel import prepare_sub_folder, write_log, get_config, DictAverageMeter, get_scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', type=str, default='configs/configs.yaml') 
    parser.add_argument('--pose_config', type=str, default='configs/train_lstm_pose.yaml') 
    
    parser.add_argument("--exp_resume", type=str, default='/hy-tmp/checkpoints/transformer_Epoch_00042000.bin', help="path to exp model checkpoint.") 
     
    parser.add_argument("--pose_resume", type=str, default="/hy-tmp/checkpoints/lstm_Epoch_00042000.bin", help="path to exp model checkpoint.") 

    parser.add_argument('--render_config', type=str, default='configs/facerender.yaml') 
    parser.add_argument('--render_ckp_backbone', type=str, default='/hy-tmp/checkpoints/facevid2vid_00189-model.pth.tar') 
    parser.add_argument('--render_ckp_mapping', type=str, default='/hy-tmp/checkpoints/mapping_00109-model.pth.tar') 

    parser.add_argument('--dataset_dir', type=str, default='/hy-tmp/test/talking_head/inputs', help="Dir of dataset root.")
    parser.add_argument('--output_dir', type=str, default='/hy-tmp/outputs/results_transformer_all_ep42000_facerender_nocrop', help="Dir to save results.")

    args = parser.parse_args()

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(args.exp_config, os.path.join(args.output_dir, 'exp_configs.yaml'))
    # shutil.copy(args.pose_config, os.path.join(args.output_dir, 'pose_configs.yaml'))
    os.makedirs(os.path.join(args.output_dir, 'preds'), exist_ok=True)

    # -- Init data loader
    dataset = InferenceDataset(dataset_dir=args.dataset_dir, subset='dev')
    dataloader = DataLoader(dataset=dataset, shuffle=False,
                            batch_size=1,
                            drop_last=False, num_workers=1, pin_memory=True)

    # -- Init model
    exp_config = get_config(args.exp_config)
    print('Using exp_config:', exp_config)
    exp_model = Transformer3DMM(**exp_config['model'])
    exp_model = exp_model.cuda()
    if args.exp_resume is not None:
        print(f'resume exp model from {args.exp_resume}. \n')
        exp_model.load_state_dict({k: v for k, v in torch.load(args.exp_resume).items()})  # dist
    exp_model.eval()

    pose_config = get_config(args.pose_config)
    pose_model = LSTMStepwiseFusion(**pose_config['model'])
    pose_model = pose_model.cuda()
    if args.pose_resume is not None:
        print(f'resume pose model from {args.pose_resume}. \n')
        pose_model.load_state_dict({k: v for k, v in torch.load(args.pose_resume).items()})  # dist
    pose_model.eval()



    # FaceVid2Vid renderer from SadTalker
    render_model = AnimateFromCoeff(free_view_checkpoint=args.render_ckp_backbone,
                                    mapping_checkpoint=args.render_ckp_mapping,
                                    config_path=args.render_config,
                                    device='cuda')

    # -- Init inference loop
    for video_idx in tqdm(range(len(dataset))):
        audio_fn = dataset.all_audio_fns[video_idx]
        # video_name = audio_fn.split('/')[-1].split('.')[-2]
        video_name = audio_fn.split('/')[-1].split('.')[0]
        print("video name:", video_name)
        data = dataset.__getitem__(video_idx)  # numpy

        first_frame = torch.from_numpy(data['first_frame']).cuda().float()  # (3, 256, 256)
        
        exp_init = torch.from_numpy(data['exp'])[:1, :].cuda().float()  # (1, 64)
        pose_init = torch.from_numpy(data['pose'])[:1, :].cuda().float()  # (1, 6)
        crop_init = torch.from_numpy(data['crop'])[:1, :].cuda().float()  # (1, 3)
        indiv_mels = torch.from_numpy(data['indiv_mels']).cuda().float()  # (num_frames, 80, 16)

        num_frames = indiv_mels.shape[0]
        # num_frames = min(indiv_mels.shape[0], len(pose)) # TODO 
        # pose = torch.from_numpy(data['pose']).cuda().float()  # (num_frames, 6) -> (1,6)
        # TODO need to change this if the head motion model is trained
        # pose = pose.repeat(num_frames,1)
        crop = torch.from_numpy(data['crop']).cuda().float()  # (num_frames, 3) -> (1,3)
        # TODO need to change this if the head motion model is trained
        crop = crop.repeat(num_frames,1)

        
        # pose = pose[:num_frames, :]
        # crop = crop[:num_frames, :]
        indiv_mels = indiv_mels[:num_frames, :, :]
        print('num_frames:', num_frames)

        with torch.no_grad():
            # -- exp_model inference
            # exp_T = exp_config['model']['embed_len']
            exp_T = 90
            # exp_T = 30
            num_iter = num_frames // exp_T + 1
            exp_preds = []
            pose_preds = []
            # crop_preds = []
            for i in range(num_iter):
                frame_indexes = list(range(i * exp_T, min(i * exp_T + exp_T, num_frames)))
                if len(frame_indexes) == 0:
                    break
                indiv_mels_clip = indiv_mels[frame_indexes][None]  # (1, exp_T, 40, 5)
                # print("indiv_mels_clip shape:", indiv_mels_clip.shape)
                # print("exp_init shape:", exp_init.shape)
                # exp_pred = exp_model(content=indiv_mels_clip, style_code=exp_init)  # (1, exp_T, 64)
                exp_pred = exp_model(content=indiv_mels_clip, init=exp_init)
                # print("exp pred shape:", exp_pred.shape)
                exp_preds.append(exp_pred)

                head_init = torch.cat((pose_init, crop_init), -1)
                # print('head init shape:', head_init.shape)
                B, T, num_mels, mel_step_size = indiv_mels_clip.shape
                audio_diven = indiv_mels_clip.view(B, T, num_mels * mel_step_size)
                driven = torch.cat((exp_pred, audio_diven), 2)
                head_pred = pose_model(init=head_init.unsqueeze(1), driven=driven)
                # print("head pred shape:", head_pred.shape)
                
                pose_pred, crop_pred = torch.split(head_pred, [6,3], -1)
                pose_preds.append(pose_pred)
                # crop_preds.append(crop_pred)

                exp_init = exp_pred[:, -1, :]  # (1, 64)
                pose_init = pose_pred[:, -1, :]
                # print("pose_init shape:", pose_init.shape)
                # crop_init = crop_pred[:, -1, :]
                
                
            
            exp_preds = torch.cat(exp_preds, dim=1)
            pose_preds = torch.cat(pose_preds, dim=1)
            # crop_preds = torch.cat(crop_preds, dim=1)

            # print('exp_preds shape', exp_preds.shape)  # (1, num_frames, 64)

            # test renderer
            # exp_preds = torch.from_numpy(data['exp']).cuda().float().unsqueeze(dim=0)

            # # -- pose_model inference
            # pose_T = 30
            # # inference steps...
            # pose_preds = pose_init.unsqueeze(dim=1).expand(-1, num_frames, -1)  # (1, num_frames, 6)
            # pose_preds = pose.unsqueeze(dim=0) # TODO
            crop_preds = crop.unsqueeze(dim=0) # TODO
            # test renderer
            # pose_preds = torch.from_numpy(data['pose']).cuda().float().unsqueeze(dim=0)

            # -- renderer inference
            source_image = data['first_frame']
            exp_init = torch.from_numpy(data['exp'])[:1, :].cuda().float() 
            pose_init = torch.from_numpy(data['pose'])[:1, :].cuda().float()  # (1, 6)
            crop_init = torch.from_numpy(data['crop'])[:1, :].cuda().float()  # (1, 3)
            # source_semantics = torch.cat([exp_init, pose_init], dim=-1).cpu().numpy()  # (1, 70)
            # generated_3dmm = torch.cat([exp_preds, pose_preds], dim=-1)[0].cpu().numpy()  # (num_frames, 70)
            source_semantics = torch.cat([exp_init, pose_init, crop_init], dim=-1).cpu().numpy()  # (1, 73)
            generated_3dmm = torch.cat([exp_preds, pose_preds, crop_preds], dim=-1)[0].cpu().numpy()  # (num_frames, 73)

            data = get_facerender_data(generated_3dmm, source_image, source_semantics, batch_size=2)
            video_pred = render_model.generate(data)

            # write to video
            print('video_pred shape:', video_pred.shape)
            video = []
            for idx in range(video_pred.shape[0]):
                image = video_pred[idx]
                image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
                video.append(image)
            result = img_as_ubyte(video)
            path = os.path.join(args.output_dir, 'preds', video_name+'.speaker.mp4')
            imageio.mimsave(path, result, fps=float(30))


