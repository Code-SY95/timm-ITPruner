import timm
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from collections import defaultdict

class FLOPsAnalyzer:
    def __init__(self, model_name, input_shape=(1, 3, 224, 224)):
        self.model_name = model_name
        self.input_shape = input_shape
        self.model = timm.create_model(model_name, pretrained=False)
        self.model.eval()
        self.graph = symbolic_trace(self.model)
        self.modules = dict(self.graph.named_modules())
        self.is_transformer = any(kw in model_name.lower() for kw in ['vit', 'swin', 'transformer', 'deit', 'mobilevit'])

        self.flops_singlecfg = defaultdict(float)
        self.flops_doublecfg = defaultdict(lambda: defaultdict(float))
        self.flops_squarecfg = defaultdict(float)
        self.node_idx_map = {}
        self.idx_counter = 0
        self.shape_cache = {}

    def _get_next_idx(self, name):
        if name not in self.node_idx_map:
            self.node_idx_map[name] = self.idx_counter
            self.idx_counter += 1
        return self.node_idx_map[name]

    def _get_tensor_shape(self, tensor):
        try:
            return tuple(tensor.shape)
        except:
            return None

    def analyze(self):
        dummy_input = torch.randn(*self.input_shape)
        with torch.no_grad():
            _ = self.model(dummy_input)

        for node in self.graph.graph.nodes:
            if node.op == 'placeholder':
                self.shape_cache[node.name] = self.input_shape
                self._get_next_idx(node.name)

            elif node.op == 'call_module':
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
                    continue  # skip modules that fail during dummy forward

                self.shape_cache[node.name] = output_shape
                idx = self._get_next_idx(node.name)
                src_idx = self._get_next_idx(input_node.name) if hasattr(input_node, 'name') else idx

                # CNN: Conv2d
                if isinstance(mod, nn.Conv2d) and not self.is_transformer:
                    C_in = mod.in_channels
                    C_out = mod.out_channels
                    K = mod.kernel_size[0]
                    H, W = output_shape[-2:]
                    flops = C_in * C_out * K * K * H * W
                    self.flops_singlecfg[idx] += flops
                    self.flops_doublecfg[src_idx][idx] += flops

                # FFN Linear or general Linear
                elif isinstance(mod, nn.Linear):
                    is_ffn = ("mlp" in node.target.lower() or "ffn" in node.target.lower())
                    if not self.is_transformer or (self.is_transformer and is_ffn):
                        in_features = mod.in_features
                        out_features = mod.out_features
                        N = output_shape[1] if output_shape and len(output_shape) > 1 else 196
                        flops = 2 * N * in_features * out_features
                        self.flops_singlecfg[idx] += flops
                        self.flops_doublecfg[src_idx][idx] += flops

            # Residual connection (e.g., torch.add)
            elif node.op == 'call_function' and node.target == torch.add:
                input_name = node.args[0].name if hasattr(node.args[0], 'name') else None
                shape = self.shape_cache.get(input_name)
                if shape and len(shape) >= 3:
                    C = shape[-1] if self.is_transformer else shape[1]
                    H = W = shape[1] if self.is_transformer else shape[2]
                    flops = C * H * W
                    idx = self._get_next_idx(node.name)
                    self.flops_squarecfg[idx] += flops
                    self.flops_singlecfg[idx] += flops
                    for arg in node.args:
                        if hasattr(arg, 'name'):
                            src = self._get_next_idx(arg.name)
                            self.flops_doublecfg[src][idx] += flops

        return self.flops_singlecfg, self.flops_doublecfg, self.flops_squarecfg
    
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
