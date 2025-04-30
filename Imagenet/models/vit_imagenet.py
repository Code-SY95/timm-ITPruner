import timm
import torch

class Vit_Imagenet(MyNetwork):
    def __init__(self, cfg=None, depth=18, block=BasicBlock, num_classes=1000):
    model = timm.create_model('vit_base_patch16_224')
    final_feat, intermediates = model.forward_intermediates(input) 
    output = model.forward_head(final_feat)  # pooling + classifier head

print(f"final feature shape : {final_feat.shape}")

print("intermediates shape :")
for f in intermediates:
    print(f.shape)
    
print(f"output shape : {output.shape}")