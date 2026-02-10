import os
import argparse
import numpy as np
import math
from scipy import optimize
import random, sys
sys.path.append("../../ITPruner")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CKA import cka
from flops_analyzer import FLOPsAnalyzer

import torch
import torch.nn as nn
import timm
from timm.models import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import datasets
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis
from typing import List, Tuple


class PrunableModelWrapper:
    def __init__(self, model_name: str, pretrained: bool = True):
        # 1. timm 모델 로드
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.eval().cuda()
        self.extract_layers()

    def extract_layers(self):
        # 2. 전체 모델 레이어 리스트 확보
        self.layer_list = [(name, module) for name, module in self.model.named_modules()]

    def print_structure(self):
        # 3. 모델 구조 출력
        for name, module in self.model.named_modules():
            print(f"{name}: {module.__class__.__name__}")

    def select_prunable_layers(self, exclude_types: Tuple[type] = (nn.Linear, nn.MultiheadAttention)):
        # 4. Attention, Classifier 제외한 레이어만 선택
        self.prunable_layers = []
        for name, module in self.layer_list:
            if isinstance(module, exclude_types):
                continue
            if isinstance(module, (nn.Conv2d, nn.ReLU, nn.GELU, nn.BatchNorm2d, nn.LayerNorm, nn.Conv1d)):
                self.prunable_layers.append((name, module))
        return self.prunable_layers

    def register_hooks(self, layers: List[Tuple[str, nn.Module]]):
        # Hook을 걸어 feature 저장
        self.features = []
        self.hooks = []

        def hook_fn(module, input, output):
            self.features.append(output)

        for _, module in layers:
            self.hooks.append(module.register_forward_hook(hook_fn))

    # def extract_features(self, x, layers=None):
    #     # 5. 선택된 레이어의 feature 추출
    #     if layers is None:
    #         layers = self.prunable_layers
    #     self.features = []
    #     self.register_hooks(layers)
    #     with torch.no_grad():
    #         _ = self.model(x)
    #     for h in self.hooks:
    #         h.remove()
    #     return self.features

    def extract_features(model_name: str, x: torch.Tensor, pretrained=True):
        model = create_model(model_name, pretrained=pretrained)
        model.eval()

        features = []
        hooks = []

        def save_output(module, input, output):
            features.append(output)

        # Transformer 여부 판별
        is_transformer = hasattr(model, 'blocks') or 'vit' in model_name.lower() or 'swin' in model_name.lower()

        if is_transformer:
            # FFN Layer 기준 Hook 등록
            if hasattr(model, 'blocks'):
                blocks = model.blocks
            elif hasattr(model, 'layers'):
                blocks = []
                for stage in model.layers:
                    blocks.extend(stage)
            else:
                raise ValueError("Transformer blocks not found")

            for block in blocks:
                if hasattr(block, 'mlp'):
                    hooks.append(block.mlp.register_forward_hook(save_output))
                elif hasattr(block, 'ffn'):
                    hooks.append(block.ffn.register_forward_hook(save_output))
        else:
            # CNN 모델 - Conv2d에 Hook 등록
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    hooks.append(module.register_forward_hook(save_output))

        # Forward 수행
        with torch.no_grad():
            _ = model(x)

        # Hook 제거
        for h in hooks:
            h.remove()

        return features

    # def compute_importance(self, x):
    #     """
    #     각 레이어의 중요도 계산
    #     """
    #     # ---------------------- 1. Feature Extraction ----------------------
    #     config = resolve_data_config({}, model=self.model)
    #     transform = create_transform(**config)
    #     dataset = datasets.ImageFolder(root=args.dataset_path, transform=transform)
        
    #     dataloader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.n_worker)
    #     data = next(iter(dataloader))
    #     data = data[0]  # (data, label)에서 data만 사용
    #     n = data.size(0)

    #     with torch.no_grad():
    #         features = self.extract_features(data.cuda())  # self.feature_extract와 동일한 역할

    #     for i in range(len(features)):
    #         features[i] = features[i].view(n, -1)
    #         features[i] = features[i].cpu().numpy()
    
    def compute_importance_with_cka(wrapper, x, beta=1.0, target_flops_ratio=0.5):
        # 5. 각 레이어의 중요도 계산
        cfg = wrapper.cfg
        length = len(cfg)

        features = wrapper.extract_features(x, wrapper.prunable_layers)
        n = x.size(0)
        for i in range(len(features)):
            features[i] = features[i].view(n, -1)
            features[i] = features[i].data.cpu().numpy()

        similar_matrix = np.zeros((len(features), len(features)))
        for i in range(len(features)):
            for j in range(len(features)):
                similar_matrix[i][j] = cka.cka(
                    cka.gram_linear(features[i]), cka.gram_linear(features[j])
                )

        def sum_list(a, j):
            return sum(a[i] for i in range(len(a)) if i != j)

        temp = [sum_list(similar_matrix[i], i) for i in range(len(features))]
        b = sum_list(temp, -1)
        temp = [x / b for x in temp]
        important = [math.exp(-1 * beta * t) for t in temp]
        important = np.negative(np.array(important))

        flops_singlecfg, flops_doublecfg, flops_squarecfg = wrapper.cfg2flops_perlayer()
        original_flops = wrapper.cfg2flops()
        target_flops = target_flops_ratio * original_flops

        def func(x, sign=1.0):
            return sum(x[i] * important[i] for i in range(length))

        def func_deriv(x, sign=1.0):
            return np.array([sign * important[i] for i in range(length)])

        def constrain_func(x):
            a = []
            for i in range(length):
                a.append(x[i] * flops_singlecfg[i])
                a.append(flops_squarecfg[i] * x[i] * x[i])
            for i in range(1, length):
                for j in range(i):
                    a.append(x[i] * x[j] * flops_doublecfg[i][j])
            return np.array([target_flops - sum(a)])

        result = optimize.minimize(
            func,
            x0=[1 for _ in range(length)],
            jac=func_deriv,
            method='SLSQP',
            bounds=[(0, 1)] * length,
            constraints={'type': 'ineq', 'fun': constrain_func}
        )

        prun_cfg = np.around(np.array(cfg) * result.x)
        optimize_cfg = [int(c) for c in prun_cfg]
        pruned_flops = wrapper.cfg2flops(cfg=optimize_cfg)

        return optimize_cfg, pruned_flops, original_flops
    
    def get_total_flops(self):
        total_flops = 0
        dummy_input = torch.randn(*self.input_shape)
        with torch.no_grad():
            _ = self.model(dummy_input)

        for node in self.graph.graph.nodes:
            if node.op == 'call_module':
                mod = self.modules[node.target]
                input_node = node.args[0]
                if hasattr(input_node, 'name') and input_node.name in self.shape_cache:
                    input_shape = self.shape_cache[input_node.name]
                else:
                    input_shape = self.input_shape

                try:
                    out_tensor = mod(torch.randn(*input_shape))
                    output_shape = self._get_tensor_shape(out_tensor)
                except:
                    continue

                self.shape_cache[node.name] = output_shape

                # CNN: Conv2d
                if isinstance(mod, nn.Conv2d) and not self.is_transformer:
                    C_in = mod.in_channels
                    C_out = mod.out_channels
                    K = mod.kernel_size[0]
                    H, W = output_shape[-2:]
                    flops = C_in * C_out * K * K * H * W
                    total_flops += flops

                # FFN Linear
                elif isinstance(mod, nn.Linear):
                    is_ffn = ("mlp" in node.target.lower() or "ffn" in node.target.lower())
                    if not self.is_transformer or (self.is_transformer and is_ffn):
                        in_features = mod.in_features
                        out_features = mod.out_features
                        N = output_shape[1] if output_shape and len(output_shape) > 1 else 196
                        flops = 2 * N * in_features * out_features
                        total_flops += flops

            # Residual connection
            elif node.op == 'call_function' and node.target == torch.add:
                input_name = node.args[0].name if hasattr(node.args[0], 'name') else None
                shape = self.shape_cache.get(input_name)
                if shape and len(shape) >= 3:
                    C = shape[-1] if self.is_transformer else shape[1]
                    H = W = shape[1] if self.is_transformer else shape[2]
                    flops = C * H * W
                    total_flops += flops

        return total_flops

    # def cfg2flops(self, input_shape=(1, 3, 224, 224)):
    #     # 6. 전체 모델의 FLOPs 계산
    #     dummy_input = torch.randn(*input_shape).cuda()
    #     flops = FlopCountAnalysis(self.model, dummy_input)
    #     return flops.total()
