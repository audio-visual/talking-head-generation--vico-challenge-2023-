# talking head generation (vico challenge 2023)
 The training and evaluation code for the vico competition ( we achived the 3rd place in the first track. Our team name: metah.)

# Conclusions may be useful for you
- If the resolution/quality of generated image is higher than the ground truth (for example, post process by gfpgan), the `FID` metric would be worse, however, the `CPBP` would be better, which is intuitive.  
  You can refer to our experiment for more details:  https://github.com/audio-visual/talking-head-generation--vico-challenge-2023-/blob/main/eval/fid_cpbd_eval.ipynb 
- A good render is very important. In our experiment, we found that adding head pose or crop can make the `LDM` and `PoseL1` better, but the moving head would cause the distort of face boundary and background, thus hampers the Visual Quality related metrics. We are very looking forward to seeing how the first and second place solutions address this issue.
- At the beginning, we made a mistake in the data of the first frame, which means that the image of the first frame may be not correspond exactly to the 3dmm coefficients. This mismatching would lead to some images not being correctly driven by the render (we only tried on face-render, for other renders the results may be different).

# Generated demos
## with head pose(only rotation) 
the left one is the collection of good results, while the right one is the collection of bad results

good results             |  bad results
:-------------------------:|:-------------------------:
<video  src="https://github.com/audio-visual/talking-head-generation--vico-challenge-2023-/assets/110716367/04d13f9b-579d-4492-ac38-1ce423f497e3" type="video/mp4"> </video>  |  <video  src="https://github.com/audio-visual/talking-head-generation--vico-challenge-2023-/assets/110716367/ffe92ecc-9403-42e7-ac27-a9ac28fbd577" type="video/mp4"> </video> 

## without head pose
the left one is the collection of good results, while the right one is the collection of bad results

good results             |  bad results
:-------------------------:|:-------------------------:
<video  src="https://github.com/audio-visual/talking-head-generation--vico-challenge-2023-/assets/110716367/6434978f-1602-4f4e-8b33-061d43e2edb4" type="video/mp4"> </video>  |  <video  src="https://github.com/audio-visual/talking-head-generation--vico-challenge-2023-/assets/110716367/62da1525-ea06-44bb-ba76-b0afd8c7b042" type="video/mp4"> </video> 
<video  src="https://github.com/audio-visual/talking-head-generation--vico-challenge-2023-/assets/110716367/80d88219-bcd1-44e5-84ca-590b93d9a470" type="video/mp4"> </video>  |  <video  src="https://github.com/audio-visual/talking-head-generation--vico-challenge-2023-/assets/110716367/9c927a25-6cfc-410d-beee-33e91423ab74" type="video/mp4"> </video> 


# Evaluation results
Actually, we propose two methods, both methods can achieve the 3rd place. 

|Method |SSIM↑ | CPBD↑	|PSNR↑|	FID↓|CSIM↑|	PoseL1↓|ExpL1↓|	AVOffset→|AVConf↑|	LipLMD↓|
|------|------|------|------|------|------|------|------|------|------|------|
|method1|0.613|	0.204	|17.811|28.829|	0.540|	0.101|	0.151|	-1.733|	2.541|	12.192|	07.01|
| method2| 0.609| 0.196	|17.579|29.184|	0.538|0.103|0.160|-0.422|1.455|12.224|07.05|


# Inference pipeline for method2 (more steps)

 ## step1
 extract keypoints and the 3dmm coefficients for the first frames 

**Note:** this step needs to prepare environment according to [deep3d_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/73d491102af6731bded9ae6b3cc7466c3b2e9e48#installation). Actually, we use two seperate enviroments, one for preprocess(deep3d_pytorch), one for others(sadtalker)
 
 Our data structure is slightly different from the baseline,   
 where the baseline is: 
 > data/train/xx.mp4(.png)  

 we do not have the train or test subfolder: 
 > data/xx.mp4(.png)  

extract facial landmarks from first frames  
 ```python
 python extract_kp_images.py \
  --input_dir ../../data/talking_head/first_frames/ \
  --output_dir ../../data/talking_head/keypoints/ \
  --device_ids 0 \
  --workers 2
 ```
 extract coefficients for first frames
 ```python
python face_recon_images.py \
  --input_dir ../../data/talking_head/first_frames/ \
  --keypoint_dir ../../data/talking_head/keypoints/ \
  --output_dir ../../data/talking_head/recons/ \
  --inference_batch_size 128 \
  --name=official \ # we rename it to official
  --epoch=20 \
  --model facerecon
 ```
 ## step2
 predicting the 3dmm coefficients for the test audios, and feed to the render
 ```python
 
 ```
 ## step3
 pass the rendered video to wav2lip
 ```python
 
 ```
 ## step4
 extract keypoints and the 3dmm coefficients for the wav2lip generated videos
```python
 
 ```
 ## step5
 re-render videos using the coefficients obtained from step4
 ```python
 
 ```
 ## step6
 combine audio and generated video
 ```python

 ```

# Training pipeline
