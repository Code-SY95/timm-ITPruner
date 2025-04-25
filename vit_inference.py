import torch
import timm
from torchvision import transforms
from PIL import Image
import urllib.request
import time

# ✅ 모델 로드
model_name = 'vit_base_patch16_clip_224.openai'
model = timm.create_model(model_name, pretrained=False)

# ✅ 저장된 weight 로드 (CLIP pretrained)
ckpt_path = '/home/sogang/mnt/db_2/oh/ITPruner/vit_base_patch16_224.pth'
state_dict = torch.load(ckpt_path, map_location='cpu')

# 일부 키 mismatch 방지
model.load_state_dict(state_dict, strict=False)
model.eval()

# ✅ 테스트용 이미지 (랜덤 또는 웹에서 로드)
# 예: 고양이 이미지
image = Image.open("Cat03.jpg").convert("RGB")

# ✅ 이미지 전처리 (CLIP 기준 224x224)
transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),  # OpenAI CLIP preprocessing mean
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

input_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

# ✅ 추론(: feature 추출) 실행 + 시간 측정
with torch.no_grad():
    start_time = time.time()  # 시작 시간
    output = model(input_tensor)
    end_time = time.time()    # 종료 시간
    
# ImageNet 클래스 인덱스 JSON 다운로드
# url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# class_names = urllib.request.urlopen(url).read().decode('utf-8').splitlines()

top5_indices = torch.topk(output, 5).indices[0].tolist()

print("🔍 모델 출력:", output.shape)
# print(output)
# print("🔥 예측 결과 (Top-5 index):", top5_indices) # 예측 결과 (Top-5 index): [190, 180, 462, 144, 343]
# # 매핑 결과 출력
# print("🔍 예측 클래스:")
# for idx in top5_indices:
#     print(f"{idx}: {class_names[idx]}")
    
# ✅ Inference 시간 출력
elapsed_time = end_time - start_time
print(f"⏱️ Inference Time: {elapsed_time:.4f}초")