import os
import torch
import numpy as np
import re
from PIL import Image
from typing import Literal
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from comfy.model_management import get_torch_device
import folder_paths

from uno.flux.modules.conditioner import HFEmbedder


# 添加自定义加载模型的函数
def custom_load_flux_model(model_path, device, use_fp8, lora_rank=512, lora_path=None):
    """
    从指定路径加载 Flux 模型
    """
    from uno.flux.model import Flux
    from uno.flux.util import load_model
    from uno.flux.util import configs, print_load_warning, set_lora
    from safetensors.torch import load_file as load_sft
    
    if use_fp8:
        params = configs["flux-dev-fp8"].params
    else:
        params = configs["flux-dev"].params
    
    # 初始化模型
    with torch.device("meta" if model_path is not None else device):
        model = Flux(params)
    
    # 如果有lora，设置 LoRA 层
    if os.path.exists(lora_path):
        print(f"Using only_lora mode with rank: {lora_rank}")
        model = set_lora(model, lora_rank, device="meta" if model_path is not None else device)
    
    # 加载模型权重
    if model_path is not None:
        print(f"Loading Flux model from {model_path}")
        print("Loading lora")
        lora_sd = load_sft(lora_path, device=str(device)) if lora_path.endswith("safetensors")\
            else torch.load(lora_path, map_location='cpu')
        print("Loading main checkpoint")
        if model_path.endswith('safetensors'):
            if use_fp8:
                print(
                    "####\n"
                    "We are in fp8 mode right now, since the fp8 checkpoint of XLabs-AI/flux-dev-fp8 seems broken\n"
                    "we convert the fp8 checkpoint on flight from bf16 checkpoint\n"
                    "If your storage is constrained"
                    "you can save the fp8 checkpoint and replace the bf16 checkpoint by yourself\n"
                )
                sd = load_sft(model_path, device="cpu")
                sd = {k: v.to(dtype=torch.float8_e4m3fn, device=device) for k, v in sd.items()}
            else:
                sd = load_sft(model_path, device=str(device))
            
            sd.update(lora_sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        else:
            dit_state = torch.load(model_path, map_location='cpu')
            sd = {}
            for k in dit_state.keys():
                sd[k.replace('module.','')] = dit_state[k]
            sd.update(lora_sd)
            missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
            model.to(str(device))
        print_load_warning(missing, unexpected)

    return model

def custom_load_ae(ae_path, device):
    """
    从指定路径加载自编码器
    """
    from uno.flux.modules.autoencoder import AutoEncoder
    from uno.flux.util import load_model
    from uno.flux.util import configs
    from safetensors.torch import load_file as load_sft
    
    # 获取对应模型类型的自编码器参数
    ae_params = configs["flux-dev"].ae_params
    
    # 初始化自编码器
    with torch.device("meta" if ae_path is not None else device):
        ae = AutoEncoder(ae_params)
    
    # 加载自编码器权重
    if ae_path is not None:
        print(f"Loading AutoEncoder from {ae_path}")
        if ae_path.endswith('safetensors'):
            sd = load_sft(ae_path, device=str(device))
        else:
            sd = torch.load(ae_path, map_location=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        if len(missing) > 0:
            print(f"Missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {len(unexpected)}")
        
        # 转移到目标设备
        ae = ae.to(str(device))
    return ae

def custom_load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    version = "xlabs-ai/xflux_text_encoders"
    if os.path.exists("/models/clip/xflux_text_encoders"):
        version = "/models/clip/xflux_text_encoders"
        return HFEmbedder(version, max_length=max_length, is_clip=False, torch_dtype=torch.bfloat16).to(device)
    cache_dir = folder_paths.get_folder_paths("clip")[0]
    return HFEmbedder(version, max_length=max_length, is_clip=False, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device)

def custom_load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    version = "openai/clip-vit-large-patch14"
    if os.path.exists("/models/clip/clip-vit-large-patch14"):
        version = "/models/clip/clip-vit-large-patch14"
        return HFEmbedder(version, max_length=77, is_clip=True, torch_dtype=torch.bfloat16).to(device)
    cache_dir = folder_paths.get_folder_paths("clip")[0]
    return HFEmbedder(version, max_length=77, is_clip=True, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device)



class REDUNOModelLoader:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "UNO_MODEL"
        self.loaded_model = None

    @classmethod
    def INPUT_TYPES(cls):
        # 获取 unet 模型列表和 vae 模型列表
        model_paths = folder_paths.get_filename_list("unet")
        vae_paths = folder_paths.get_filename_list("vae")
        
        # 增加 LoRA 模型选项
        lora_paths = folder_paths.get_filename_list("loras")
        
        return {
            "required": {
                "flux_model": (model_paths, ),
                "ae_model": (vae_paths, ),
                "use_fp8": ("BOOLEAN", {"default": False}),
                "offload": ("BOOLEAN", {"default": False}),
                "lora_model": (["None"] + lora_paths, ),
            }
        }

    RETURN_TYPES = ("UNO_MODEL",)
    RETURN_NAMES = ("uno_model",)
    FUNCTION = "load_model"
    CATEGORY = "UNO"

    def load_model(self, flux_model, ae_model, use_fp8, offload, lora_model=None):
        device = get_torch_device()
        from uno.flux.pipeline import UNOPipeline
        
        try:
            # 获取模型文件的完整路径
            flux_model_path = folder_paths.get_full_path("unet", flux_model)
            ae_model_path = folder_paths.get_full_path("vae", ae_model)
            
            # 获取LoRA模型路径（如果有）
            lora_model_path = None
            if lora_model is not None and lora_model != "None":
                lora_model_path = folder_paths.get_full_path("loras", lora_model)
            
            print(f"Loading Flux model from: {flux_model_path}")
            print(f"Loading AE model from: {ae_model_path}")
            lora_rank = 512
            if lora_model_path:
                print(f"Loading LoRA model from: {lora_model_path}")
            
            # 创建自定义 UNO Pipeline
            class CustomUNOPipeline(UNOPipeline):
                def __init__(self, use_fp8, device, flux_path, ae_path, offload=False, 
                            lora_rank=512, lora_path=None):
                    self.device = device
                    self.offload = offload
                    self.model_type = "flux-dev-fp8" if use_fp8 else "flux-dev"
                    self.use_fp8 = use_fp8
                    # 加载 CLIP 和 T5 编码器
                    self.clip = custom_load_clip(device="cpu" if offload else self.device)
                    self.t5 = custom_load_t5(device="cpu" if offload else self.device, max_length=512)
                    
                    # 加载自定义模型
                    self.ae = custom_load_ae(ae_path, device="cpu" if offload else self.device)
                    self.model = custom_load_flux_model(
                        flux_path, 
                        device="cpu" if offload else self.device, 
                        use_fp8=use_fp8,
                        lora_rank=lora_rank,
                        lora_path=lora_path
                    )
                    
            # 创建自定义 pipeline
            model = CustomUNOPipeline(
                use_fp8=use_fp8,
                device=device,
                flux_path=flux_model_path,
                ae_path=ae_model_path,
                offload=offload,
                lora_rank=lora_rank,
                lora_path=lora_model_path,
            )
            
            self.loaded_model = model
            print(f"UNO model loaded successfully with custom models.")
            return (model,)
        except Exception as e:
            print(f"Error loading UNO model: {e}")
            import traceback
            traceback.print_exc()
            raise e


class REDUNOGenerate:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uno_model": ("UNO_MODEL",),
                "prompt": ("STRING", {"multiline": True}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 16}),
                "guidance": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 3407}),
                "pe": (["d", "h", "w", "o"], {"default": "d"}),
            },
            "optional": {
                "reference_image_1": ("IMAGE",),
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "UNO"

    def generate(self, uno_model, prompt, width, height, guidance, num_steps, seed, pe, 
                reference_image_1=None, reference_image_2=None, reference_image_3=None, reference_image_4=None):
        # Make sure width and height are multiples of 16
        width = (width // 16) * 16
        height = (height // 16) * 16
        from uno.flux.pipeline import preprocess_ref
        
        # Process reference images if provided
        ref_imgs = []
        ref_tensors = [reference_image_1, reference_image_2, reference_image_3, reference_image_4]
        for ref_tensor in ref_tensors:
            if ref_tensor is not None:
                # Convert from tensor to PIL
                if isinstance(ref_tensor, torch.Tensor):
                    # Handle batch of images
                    if ref_tensor.dim() == 4:  # [batch, height, width, channels]
                        for i in range(ref_tensor.shape[0]):
                            img = ref_tensor[i].cpu().numpy()
                            ref_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                            # Determine reference size based on number of reference images
                            ref_size = 512 if len([t for t in ref_tensors if t is not None]) <= 1 else 320
                            ref_image_pil = preprocess_ref(ref_image_pil, ref_size)
                            ref_imgs.append(ref_image_pil)
                    else:  # [height, width, channels]
                        img = ref_tensor.cpu().numpy()
                        ref_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                        # Determine reference size based on number of reference images
                        ref_size = 512 if len([t for t in ref_tensors if t is not None]) <= 1 else 320
                        ref_image_pil = preprocess_ref(ref_image_pil, ref_size)
                        ref_imgs.append(ref_image_pil)
                elif isinstance(ref_tensor, np.ndarray):
                    # Assume ComfyUI range is [-1, 1], convert to [0, 1]
                    ref_image_pil = Image.fromarray((img * 255).astype(np.uint8))
                    # Determine reference size based on number of reference images
                    ref_size = 512 if len([t for t in ref_tensors if t is not None]) <= 1 else 320
                    ref_image_pil = preprocess_ref(ref_image_pil, ref_size)
                    ref_imgs.append(ref_image_pil)
        
        try:
            # Generate image
            output_img = uno_model(
                prompt=prompt,
                width=width,
                height=height,
                guidance=guidance,
                num_steps=num_steps,
                seed=seed,
                ref_imgs=ref_imgs,
                pe=pe
            )
            
            # Save the generated image
            output_filename = f"uno_{seed}_{prompt[:20].replace(' ', '_')}.png"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Convert to ComfyUI-compatible tensor
            if hasattr(output_img, 'images') and len(output_img.images) > 0:
                # Handle FluxPipelineOutput
                output_img.images[0].save(output_path)
                print(f"Saved UNO generated image to {output_path}")
                image = np.array(output_img.images[0]) / 255.0  # Convert to [0, 1]
            else:
                # Handle PIL Image
                output_img.save(output_path)
                print(f"Saved UNO generated image to {output_path}")
                image = np.array(output_img) / 255.0  # Convert to [0, 1]
            
            # Convert numpy array to torch.Tensor
            image = torch.from_numpy(image).float()
            
            # Make sure it's in ComfyUI format [batch, height, width, channels]
            if image.dim() == 3:  # [height, width, channels]
                image = image.unsqueeze(0)  # Add batch dimension to make it [1, height, width, channels]
            
            
            return (image,)
        except Exception as e:
            print(f"Error generating image with UNO: {e}")
            raise e


# Register our nodes to be used in ComfyUI
NODE_CLASS_MAPPINGS = {
    "REDUNOModelLoader": REDUNOModelLoader,
    "REDUNOGenerate": REDUNOGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "REDUNOModelLoader": "UNO Model Loader @REDAIGC",
    "REDUNOGenerate": "UNO Generate @REDAIGC",
}
