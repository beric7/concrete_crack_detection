""" DeepLabv3 Model download and change the head for your prediction"""
from network._deeplab import DeepLabHeadV3Plus # good
from network.modeling import deeplabv3plus_resnet50 # good
from network.modeling import deeplabv3plus_resnet101 # good
# from torchvision import models
# n_classes is the number of probabilities you want to get per pixel
#   - For 1 class and background, use n_classes=1
#   - For 2 classes, use n_classes=1
#   - For N > 2 classes, use n_classes=N

def createDeepLabv3Plus(outputchannels=4, output_stride=8):
    # See what happens if I set pre-trained equal to True vs. False...
    # If the model is set to pre-trained, then we will use a defined model weight
    # These model weights must be currently manually changed in:
    # models/segmentation/segmentation.py
    inplanes = 2048
    low_level_planes = 256
    num_classes = outputchannels
    model = deeplabv3plus_resnet101(num_classes=num_classes, output_stride=8)
    model.classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes)
    # Set the model in training mode
    model.train()
    return model
