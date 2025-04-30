import argparse
import os
import torch
import torch.nn as nn

# (필요 시) DataLoader 및 Transform 관련 import
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import datasets
from torch.utils.data import DataLoader

# ---------------------
# Argument Parsing
# ---------------------
parser = argparse.ArgumentParser(description="Feature Extraction with ITPruner")
# parser.add_argument('--use_timm', action='store_true', help='Use timm to load model')
parser.add_argument('--model', type=str, required=True, help='Model name (e.g., resnet50, efficientnet_b0 for timm)')
parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset name')
parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--save_path', type=str, default='/pruned_model')
args = parser.parse_args()

# ---------------------
# Load Model
# ---------------------
import timm
model = timm.create_model(args.model, pretrained=True)

model.eval()
model.cuda()

# ---------------------
# Dataset & Transform
# ---------------------
# if args.use_timm:
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
# else:
#     from torchvision import transforms
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ])

dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# ---------------------
# Feature Extraction Example
# ---------------------
with torch.no_grad():
    for images, labels in dataloader:
        images = images.cuda()
        features = model(images)
        
        # TODO: Save or process extracted features
        print(f"Extracted features shape: {features.shape}")
        break  # 예시: 한 배치만 처리

