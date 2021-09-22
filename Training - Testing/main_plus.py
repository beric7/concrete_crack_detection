import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, jaccard_score
from model_plus import createDeepLabv3Plus

import sys
print(sys.version, sys.platform, sys.executable)
from trainer_plus import train_model
import datahandler_plus
import argparse
import os
import torch
torch.cuda.empty_cache()

"""
    Version requirements:
        PyTorch Version:  1.2.0
        Torchvision Version:  0.4.0a0+6b959ee
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "-data_directory", help='Specify the dataset directory path')
parser.add_argument(
    "-exp_directory", help='Specify the experiment directory where metrics and model weights shall be stored.')
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--batchsize", default=2, type=int)
parser.add_argument("--output_stride", default=8, type=int)
parser.add_argument("--channels", default=2, type=int)
parser.add_argument("--pretrained", default='')

args = parser.parse_args()

bpath = args.exp_directory
data_dir = args.data_directory
epochs = args.epochs
batchsize = args.batchsize
output_stride = args.output_stride
channels = args.channels
model_path = args.pretrained
print('loading pre-trained model from saved state: ' + args.pretrained)
if model_path != '':
    try:
        model = torch.load(model_path)
        print('LOADED MODEL')
    except:
        print('model path did not load')
        model = createDeepLabv3Plus(outputchannels=channels, output_stride=output_stride)

    model = createDeepLabv3Plus(outputchannels=channels, output_stride=output_stride)
    
model.train()
# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)

# Specify the loss function
criterion = torch.nn.CrossEntropyLoss()
# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Specify the evalutation metrics
metrics = {'f1_score': f1_score, 'jaccard_score': jaccard_score}

# Create the dataloader
dataloaders = datahandler_plus.get_dataloader_sep_folder(data_dir, batch_size=batchsize)
trained_model = train_model(model, criterion, dataloaders,
                            optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)

# Save the trained model
# torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
torch.save(model, os.path.join(bpath, 'weights.pt'))
