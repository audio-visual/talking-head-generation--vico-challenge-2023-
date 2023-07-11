# talking head generation (vico challenge 2023)
 The training and evaluation code for the vico competition ( we achived the 3rd place in the first track. Our team name: metah.)

# Conclusions may be useful for you
- If the resolution/quality of generated image is higher than the ground truth (for example, post process by gfpgan), the `FID` metric would be worse, however, the `CPBP` would be better, which is intuitive.  
  You can refer to our experiment for more details:  https://github.com/audio-visual/talking-head-generation--vico-challenge-2023-/blob/main/eval/fid_cpbd_eval.ipynb 
- A good render is very important. In our experiment, we found that adding head pose or crop can make the `LDM` and `PoseL1` better, but the moving head would cause the distort of face boundary and background, thus hampers the Visual Quality related metrics. We are very looking forward to seeing how the first and second place solutions address this issue.
- At the beginning, we made a mistake in the data of the first frame, which means that the image of the first frame may be not correspond exactly to the 3dmm coefficient. This mismatching would lead to some images not being correctly driven by the render (we only tried on face-render, for other renders the results may be different).

# Generated demos
## with head pose and crop
the left one is the collection of good results, while the right one is the collection of bad results

good results             |  bad results
:-------------------------:|:-------------------------:
<video  src="https://github.com/audio-visual/talking-head-generation--vico-challenge-2023-/assets/110716367/b0260ff7-d210-42a9-9446-8285d1a9fa6e" type="video/mp4"> </video>  |  <video  src="https://github.com/audio-visual/talking-head-generation--vico-challenge-2023-/assets/110716367/814521b3-1904-4956-a466-9fa62bcddc94" type="video/mp4"> </video> 

## without head pose
the left one is the collection of good results, while the right one is the collection of bad results

# Evaluation results
Actually, we propose two methods, both methods can achieve the 3rd place. 

|Method |SSIM↑ | CPBD↑	|PSNR↑|	FID↓|CSIM↑|	PoseL1↓|ExpL1↓|	AVOffset→|AVConf↑|	LipLMD↓|
|------|------|------|------|------|------|------|------|------|------|------|
| method1| 0.609| 0.196	|17.579|29.184|	0.538|0.103|0.160|-0.422|1.455|12.224|07.05|
|method2|0.613|	0.204	|17.811|28.829|	0.540|	0.101|	0.151|	-1.733|	2.541|	12.192|	07.01|


# Inference pipeline for method1 (simpler)

 ## step1
 **extract keypoints and the 3dmm coefficients for the first frames** 
 xx
 ## step2
 **predicting the 3dmm coefficients for the test audios, and feed to the render**
 xx
 ## step3
 **combine audio and generated video**
 xx

# Inference pipeline for method2 (more steps)

 ## step1
 **extract keypoints and the 3dmm coefficients for the first frames** 
 xx
 ## step2
 **predicting the 3dmm coefficients for the test audios, and feed to the render**
 xx
 ## step3
 **pass the rendered video to wav2lip**
 xx
 ## step4
 **extract keypoints and the 3dmm coefficients for the wav2lip generated videos**
 xx
 ## step5
 **re-render videos using the coefficients obtained from step4**
 xx
 ## step6
 **combine audio and generated video**
 xx

# Training pipeline
