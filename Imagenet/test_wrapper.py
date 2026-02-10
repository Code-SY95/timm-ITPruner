from prunable_model_wrapper import PrunableModelWrapper

model_name = "vit_base_patch16_224.dino"
wrapper = PrunableModelWrapper(model_name=model_name, pretrained=True)

print(wrapper.compute_importance_with_cka())