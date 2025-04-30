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

# -------------------------------
# Environment Setup
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# -------------------------------
# Model Load
# -------------------------------
model = timm.create_model(args.model, pretrained=True)
model = model.to(device)
model.eval()

# -------------------------------
# Dataset
# -------------------------------
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

dataset = datasets.ImageFolder(root=args.dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.n_worker)

# -------------------------------
# Feature Extraction with Hook
# -------------------------------

# feature_outputs = OrderedDict()

# def hook_fn(module, input, output):
#     feature_outputs[module_name] = output

# # Hook 등록
# hooks = []

# if args.target_layers is None:
#     # 전체 레이어 대상
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.Conv2d, nn.Linear)):  # 예시: Conv나 Linear만 hook
#             module_name = name
#             hooks.append(module.register_forward_hook(lambda m, i, o: feature_outputs.setdefault(module_name, o)))
# else:
#     # 특정 레이어만 대상으로
#     for name, module in model.named_modules():
#         if name in args.target_layers:
#             module_name = name
#             hooks.append(module.register_forward_hook(lambda m, i, o: feature_outputs.setdefault(module_name, o)))

def register_hooks(model, target_layers=None):
    """
    모델에 hook을 등록해서 feature map을 캡처하는 함수
    Args:
        model: timm 모델
        target_layers: (list or None) 특정 layer name 목록. None이면 자동 선택.
    Returns:
        feature_outputs: Hook을 통해 저장된 feature dict
        hooks: 등록된 hook 핸들러 리스트 (나중에 제거용)
    """
    feature_outputs = OrderedDict()
    hooks = []

    def hook_fn(module, input, output):
        # output을 feature_outputs에 저장
        feature_outputs[module_name] = output

    for name, module in model.named_modules():
        if target_layers is None:
            # 자동 선택 모드
            if is_suitable_for_hook(name, module):
                module_name = name
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, n=module_name: feature_outputs.setdefault(n, o)
                ))
        else:
            # 사용자가 직접 고른 레이어만 Hook
            if name in target_layers:
                module_name = name
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, n=module_name: feature_outputs.setdefault(n, o)
                ))

    return feature_outputs, hooks

def is_suitable_for_hook(name, module):
    """
    이 module이 feature 추출 대상으로 적합한지 판단하는 함수
    ResNet과 ViT 둘 다 대응 가능하게
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return True
    if 'blocks' in name or 'norm' in name or 'head' in name:
        return True
    return False


# -------------------------------
# Forward Pass
# -------------------------------
data_iter = iter(dataloader)
images, _ = next(data_iter)
images = images.to(device)

feature_outputs, hooks = register_hooks(model, target_layers=args.target_layers)

with torch.no_grad():
    _ = model(images)

# Hook 제거
for h in hooks:
    h.remove()

# -------------------------------
# Features 정리
# -------------------------------
features = []
for k in sorted(feature_outputs.keys()):
    f = feature_outputs[k]
    f = f.reshape(f.size(0), -1).cpu().numpy()
    features.append(f)

n_layers = len(features)

# -------------------------------
# Similarity Matrix (CKA)
# -------------------------------
similarity_matrix = np.zeros((n_layers, n_layers))

for i in range(n_layers):
    for j in range(n_layers):
        similarity_matrix[i][j] = cka.cka(cka.gram_linear(features[i]), cka.gram_linear(features[j]))

def sum_list(a, j):
    return sum(a[i] for i in range(len(a)) if i != j)

# -------------------------------
# Importance Calculation
# -------------------------------
temp = [sum_list(similarity_matrix[i], i) for i in range(len(features))]
b = sum(temp)
temp = [x / b for x in temp]
important = np.array([math.exp(-1 * args.beta * x) for x in temp])
important = np.negative(important)

# -------------------------------
# FLOPs Dummy 값
# -------------------------------
flops_singlecfg = np.ones(n_layers)
flops_doublecfg = np.zeros((n_layers, n_layers))
flops_squarecfg = np.zeros(n_layers)

# -------------------------------
# Optimization
# -------------------------------
def objective_func(x):
    return np.sum(x * important)

def derivative_func(x):
    return important

def constraint_func(x):
    total_flops = np.sum(x * flops_singlecfg) + np.sum(x**2 * flops_squarecfg)
    for i in range(1, n_layers):
        for j in range(i):
            total_flops += x[i] * x[j] * flops_doublecfg[i][j]
    return np.array([args.target_flops - total_flops])

bounds = tuple((0, 1) for _ in range(n_layers))
constraints = ({'type': 'ineq', 'fun': constraint_func})

result = optimize.minimize(objective_func, x0=np.ones(n_layers), jac=derivative_func,
                            method='SLSQP', bounds=bounds, constraints=constraints)

# -------------------------------
# 결과 출력
# -------------------------------
pruned_cfg = result.x
print("Optimized pruning cfg:", pruned_cfg)
print("Sum of selected ratio:", np.sum(pruned_cfg))