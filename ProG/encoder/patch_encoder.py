import sys
import torch
import torch.nn as nn

from ProG.encoder.resnet import resnet50_baseline, resnet18_baseline

# for using PathoDuet
sys.path.append('/mnt/data/smart_health_02/lujiaxuan/workingplace/GleasonGrade/code/PathoDuet')
from vits import VisionTransformerMoCo

def PatchEncoder(encoder_name='Pathoduet', checkpoint_path=None):
    model = None
    if encoder_name == 'ResNet50':
        model = resnet50_baseline(pretrained=True)
    elif encoder_name == 'ResNet18':
        model = resnet18_baseline(pretrained=True)
    elif encoder_name == 'Pathoduet':
        target_patch_size = 224  # the input for the PathoDuet model
        model = VisionTransformerMoCo(pretext_token=True, global_pool='avg')
        # model.head = nn.Linear(768, args.num_classes)
        model.head = nn.Identity()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint, strict=False)

    # turn the grad off
    for param in model.parameters():
        param.requires_grad = False

    return model