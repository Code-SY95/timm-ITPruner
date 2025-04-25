import timm
import torch

model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
