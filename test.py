import timm
import torch
from torchinfo import summary

model = timm.create_model('vit_base_patch16_224')
model.eval()  # inference 모드 설정

# 올바른 입력 정의 (Batch=1, Channel=3, Height=224, Width=224)
input = torch.randn(1, 3, 224, 224)


#####################중간 단계 shape 확인#######################
# final_feat, intermediates = model.forward_intermediates(input) 
# output = model.forward_head(final_feat)  # pooling + classifier head

# print(f"final feature shape : {final_feat.shape}")

# print("intermediates shape :")
# for f in intermediates:
#     print(f.shape)
    
# print(f"output shape : {output.shape}")

#############################################################
# # 중간 레이어 인덱스 지정 (0~11)
# layer_indices = [0, 3, 6, 9, 11]

# # 중간 레이어 결과 추출
# intermediate_outputs = model.get_intermediate_layers(input, n=layer_indices)

# # 레이어 이름과 결과 매핑
# print("Layer Name\t\tOutput Shape")
# for idx, (layer_idx, feat) in enumerate(zip(layer_indices, intermediate_outputs)):
#     layer_name = f"blocks.{layer_idx} ({model.blocks[layer_idx].__class__.__name__})"
#     print(f"{layer_name:<20}\t{feat.shape}")

#############################모델 구조 요약#################################
# 2. 전체 모델 구조 출력
print("="*50 + "\nFull Model Architecture:\n" + "="*50)
print(model)

# 3. 블록별 상세 구조 출력
print("\n" + "="*50 + "\nDetailed Block Structure:\n" + "="*50)
for i, block in enumerate(model.blocks):
    print(f"\nBlock {i}:")
    print(block)
    
# 4. 레이어 요약 정보 (선택사항)
print("\n" + "="*50 + "\nLayer Summary:\n" + "="*50)
summary(model, input_size=(1, 3, 224, 224), depth=4, device="cpu")