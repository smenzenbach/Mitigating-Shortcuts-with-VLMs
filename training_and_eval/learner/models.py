import torch.nn as nn
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights


def VGG16_pretrained(freeze_features=False):
    """
    Loads the VGG16 model pretrained on ImageNet.
    Replaces the last layer of the classifier to output 2 classes instead of 1000.

    From pytorch:
        Expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images
        of shape (3 x H x W), where H and W are expected to be at least 224. The images have
        to be loaded in to a range of [0,1] and then normalized using
        mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    Args:
        freeze_features: if True then freezes all feature layers. Used to train model only
            on classifier i.e. transfer learning.
    """
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[-1] = nn.Linear(4096, 2)
    return model
