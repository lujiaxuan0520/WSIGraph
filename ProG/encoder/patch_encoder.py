import sys
import torch
import torch.nn as nn
import timm

# from ProG.encoder.resnet import resnet50_baseline, resnet18_baseline

# for using PathoDuet
sys.path.append('/mnt/hwfile/smart_health/lujiaxuan/PathoDuet')
from vits import VisionTransformerMoCo

# for using UNI
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login


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
    elif encoder_name == 'GigaPath':
        model = timm.create_model(model_name='vit_giant_patch14_dinov2', 
                **{'img_size': 224, 'in_chans': 3, 
                'patch_size': 16, 'embed_dim': 1536, 
                'depth': 40, 'num_heads': 24, 'init_values': 1e-05, 
                'mlp_ratio': 5.33334, 'num_classes': 0})
        # state_dict = torch.load('checkpoint_path', map_location='cpu')
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict, strict=True)
    elif encoder_name in ['UNI', 'PathOrchestra']:
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        model.load_state_dict(torch.load(checkpoint_path), strict=True)

    # turn the grad off
    for param in model.parameters():
        param.requires_grad = False

    return model


# if __name__ == '__main__':
    

    # login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

    # # pretrained=True needed to load UNI weights (and download weights for the first time)
    # # init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
    # model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    # transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    # model.eval()