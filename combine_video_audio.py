import subprocess
import glob 
import os 
# pred_video_folder = '/hy-tmp/outputs/results_transformer_ep38000_facerender_nopose_offset0/preds'
pred_video_folder = '/hy-tmp/outputs/results_transformer_ep42000_wav2lip_postprocess/wav2lip_gan_facerender/preds'
audio_folder = '/hy-tmp/test/talking_head/inputs/audios'
output_folder = '/hy-tmp/outputs/results_transformer_ep42000_wav2lip_postprocess/wav2lip_gan_facerender/combine'
os.makedirs(output_folder, exist_ok=True)
pred_videos = glob.glob(os.path.join(pred_video_folder,'*.mp4'))
for pred_video in pred_videos:
    name = pred_video.split(os.path.sep)[-1].split('.')[0]
    print(name)
    audio_path = os.path.join(audio_folder,name+'.wav')
    output_path = os.path.join(output_folder,name+'.speaker'+'.mp4')
    command = ("ffmpeg -i %s -i %s -c:v copy -c:a aac %s" % (pred_video, audio_path, output_path))
    output = subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print('done')
    # https://vico.solutions/leaderboard/2023