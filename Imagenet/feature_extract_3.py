import os
import argparse
import numpy as np
import math
from scipy import optimize
import random, sys
sys.path.append("../../ITPruner")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import datasets
from torch.utils.data import DataLoader
from collections import OrderedDict
from CKA import cka
import timm

parser = argparse.ArgumentParser()

""" model config """
parser.add_argument('--model', type=str, required=True,
                    help="timm model name, e.g., 'resnet50', 'vit_base_patch16_224'")
parser.add_argument('--target_layers', type=str, nargs='+', default=None,
                    help="List of layer names to target, e.g., 'layer3.5.conv2' (default: all layers)")
parser.add_argument('--target_flops', default=0, type=int)
parser.add_argument('--beta', default=1, type=int)

""" dataset config """
parser.add_argument('--dataset_path', type=str, required=True, help='Path to ImageFolder dataset')
parser.add_argument('--save_path', type=str, default='/pruned_model')

""" runtime config """
parser.add_argument('--gpu', default='0', help='GPU id to use')
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--n_worker', type=int, default=4)
parser.add_argument("--local_rank", default=0, type=int)

args = parser.parse_args()

# -----------------------------------------
# Environment Setup
# -----------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


if __name__ == '__main__':
    args = parser.parse_args()
    
    # -----------------------------------------
    # Model Load
    # -----------------------------------------
    model = timm.create_model(args.model, pretrained=True)
    model = model.to(device)
    model.eval()

    # -----------------------------------------
    # Dataset
    # -----------------------------------------
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    dataset = datasets.ImageFolder(root=args.dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.n_worker)
    
    # -----------------------------------------
    # Feature Extract
    # -----------------------------------------
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.to(device)
    
    with torch.no_grad():
        
