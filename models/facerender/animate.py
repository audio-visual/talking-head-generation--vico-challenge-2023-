import yaml
import warnings
warnings.filterwarnings('ignore')

# import sys
# sys.path.append("../../")

import torch
from models.facerender.modules.keypoint_detector import HEEstimator, KPDetector
from models.facerender.modules.mapping import MappingNet
from models.facerender.modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from models.facerender.modules.make_animation import make_animation


class AnimateFromCoeff():

    def __init__(self, free_view_checkpoint, mapping_checkpoint,
                   config_path, device):

        with open(config_path) as f:
            config = yaml.safe_load(f)

        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                    **config['model_params']['common_params'])
        kp_extractor = KPDetector(**config['model_params']['kp_detector_params'],
                                    **config['model_params']['common_params'])
        he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
        mapping = MappingNet(**config['model_params']['mapping_params'])


        generator.to(device)
        kp_extractor.to(device)
        he_estimator.to(device)
        mapping.to(device)
        for param in generator.parameters():
            param.requires_grad = False
        for param in kp_extractor.parameters():
            param.requires_grad = False 
        for param in he_estimator.parameters():
            param.requires_grad = False
        for param in mapping.parameters():
            param.requires_grad = False

        if free_view_checkpoint is not None:
            self.load_cpk_facevid2vid(free_view_checkpoint, kp_detector=kp_extractor, generator=generator, he_estimator=he_estimator)
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.")

        if  mapping_checkpoint is not None:
            self.load_cpk_mapping(mapping_checkpoint, mapping=mapping)
            print("loaded mapping checkpoint")
        else:
            raise AttributeError("Checkpoint should be specified for video head pose estimator.") 

        self.kp_extractor = kp_extractor
        self.generator = generator
        self.he_estimator = he_estimator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.he_estimator.eval()
        self.mapping.eval()
         
        self.device = device
    
    def load_cpk_facevid2vid(self, checkpoint_path, generator=None, discriminator=None, 
                        kp_detector=None, he_estimator=None, optimizer_generator=None, 
                        optimizer_discriminator=None, optimizer_kp_detector=None, 
                        optimizer_he_estimator=None, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        print("loaded face2face checkpoint")
        if generator is not None:
            generator.load_state_dict(checkpoint['generator'])
        if kp_detector is not None:
            kp_detector.load_state_dict(checkpoint['kp_detector'])
        if he_estimator is not None:
            he_estimator.load_state_dict(checkpoint['he_estimator'])
        if discriminator is not None:
            try:
               discriminator.load_state_dict(checkpoint['discriminator'])
            except:
               print ('No discriminator in the state-dict. Dicriminator will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except RuntimeError as e:
                print ('No discriminator optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
        if optimizer_he_estimator is not None:
            optimizer_he_estimator.load_state_dict(checkpoint['optimizer_he_estimator'])

        return checkpoint['epoch']
    
    def load_cpk_mapping(self, checkpoint_path, mapping=None, discriminator=None,
                 optimizer_mapping=None, optimizer_discriminator=None, device='cpu'):
        checkpoint = torch.load(checkpoint_path,  map_location=torch.device(device))
        if mapping is not None:
            mapping.load_state_dict(checkpoint['mapping'])
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'])
        if optimizer_mapping is not None:
            optimizer_mapping.load_state_dict(checkpoint['optimizer_mapping'])
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])

        return checkpoint['epoch']

    def generate(self, x):
        source_image = x['source_image'].type(torch.FloatTensor)
        source_semantics = x['source_semantics'].type(torch.FloatTensor)
        target_semantics = x['target_semantics_list'].type(torch.FloatTensor)
        source_image = source_image.to(self.device)
        source_semantics = source_semantics.to(self.device)
        target_semantics = target_semantics.to(self.device)

        frame_num = x['frame_num']

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                           self.generator, self.kp_extractor, self.he_estimator, self.mapping,
                                           yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None, use_exp=True)

        predictions_video = predictions_video.reshape((-1,)+predictions_video.shape[2:])
        predictions_video = predictions_video[:frame_num]
        return predictions_video
    
        

