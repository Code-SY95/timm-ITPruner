import torch
print(torch.cuda.get_device_properties(0).major,  # 8 (RTX 3060)
      torch.cuda.get_device_properties(0).minor)  # 6
print(torch.cuda.get_arch_list())  # sm_86 포함 여부 확인
print(torch.zeros(1).cuda())  # 간단한 텐서 생성 테스트
