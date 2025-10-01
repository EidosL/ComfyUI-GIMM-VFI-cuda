import os
import sys
import requests
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from contextlib import nullcontext
import numpy as np
import comfy.model_management as mm
import comfy.utils
import torch
import yaml

from .gimmvfi.generalizable_INR.gimmvfi_f import GIMMVFI_F
from .gimmvfi.generalizable_INR.gimmvfi_r import GIMMVFI_R
from .gimmvfi.utils.utils import InputPadder, flow_to_image

# 🗃️ 架构笔记: 增加Hugging Face自动下载功能
def hf_download(model_name, model_path):
    if os.path.exists(model_path):
        return

    # 构建Hugging Face Hub的直接下载URL
    hf_url = f"https://huggingface.co/GSean/GIMM-VFI/resolve/main/{model_name}"
    print(f"GIMM-VFI: Model not found locally. Downloading from {hf_url}...")

    try:
        response = requests.get(hf_url, stream=True)
        response.raise_for_status() # 确保请求成功

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        with open(model_path, 'wb') as file, tqdm(
            desc=model_name,
            total=total_size_in_bytes,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        
        if total_size_in_bytes != 0 and bar.n != total_size_in_bytes:
            raise IOError("ERROR, something went wrong during download")

        print(f"GIMM-VFI: Download complete.")
    except Exception as e:
        # 如果下载失败，删除可能已创建的不完整文件
        if os.path.exists(model_path):
            os.remove(model_path)
        raise ConnectionError(f"Failed to download model from Hugging Face. Please download it manually and place it in the 'checkpoints' folder. Reason: {e}")


def preflight_check(model, precision, images, interpolation_factor):
    # ... (Preflight check logic remains unchanged)
    device = mm.get_torch_device()
    if not torch.cuda.is_available():
        return torch.float32, True, 1

    props = torch.cuda.get_device_properties(device)
    
    final_dtype = precision
    if precision == torch.float16 and props.major < 7:
        final_dtype = torch.float32
    if precision == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        final_dtype = torch.float32

    bytes_per_element = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(final_dtype, 4)
    num_input_frames, h, w, c = images.shape
    
    model_vram_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    vram_per_pair_mb = (2 * h * w * c * bytes_per_element * interpolation_factor * 2.5) / (1024**2)
    
    torch.cuda.empty_cache()
    free_vram_mb = mm.get_free_memory(device) / (1024**2)
    
    available_for_batching_mb = free_vram_mb - model_vram_mb * 1.2
    if vram_per_pair_mb <= 0: vram_per_pair_mb = 1
    
    optimal_batch_size = int(available_for_batching_mb / vram_per_pair_mb)
    optimal_batch_size = max(1, min(optimal_batch_size, 16))
    
    print(f"GIMM-VFI Preflight: Auto-configured Optimal Batch Size: {optimal_batch_size}")

    total_output_frames = (num_input_frames - 1) * interpolation_factor + 1
    estimated_total_vram_mb = (total_output_frames * h * w * c * bytes_per_element) / (1024 ** 2)
    required_vram_mb = estimated_total_vram_mb + model_vram_mb * 1.5
    
    use_safe_mode = False
    if required_vram_mb > free_vram_mb:
        print(f"GIMM-VFI Warning: Estimated total VRAM ({required_vram_mb:.2f} MB) exceeds available VRAM. Activating SAFE MODE.")
        use_safe_mode = True
        
    return final_dtype, use_safe_mode, optimal_batch_size

class GIMMVFILoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": ([
                "gimmvfi_f_arb_lpips.pth", # LPIPS version as recommended default
                "gimmvfi_r_arb.pth",
                "gimmvfi_f_arb.pth",
            ],),
            "precision": (["fp32", "fp16", "bfloat16"], {"default": "fp16"}), 
            "torch_compile": ("BOOLEAN", {"default": False}),
        }}
    RETURN_TYPES = ("GIMMVFI_MODEL",)
    FUNCTION = "loadmodel"
    CATEGORY = "GIMM-VFI"

    def loadmodel(self, model_name, precision="fp16", torch_compile=False):
        try:
            checkpoints_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True) # 确保目录存在
            model_path = os.path.join(checkpoints_dir, model_name)

            # 🗃️ 架构笔记: 在加载前调用下载函数
            hf_download(model_name, model_path)

            # 根据模型名选择配置文件
            if "gimmvfi_f" in model_name:
                config_name = "gimmvfi_f_arb.yaml"
            elif "gimmvfi_r" in model_name:
                config_name = "gimmvfi_r_arb.yaml"
            else: # 默认
                config_name = "gimmvfi_f_arb.yaml"

            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs/gimmvfi", config_name)
            with open(config_path, "r") as f: config = yaml.load(f, Loader=yaml.FullLoader)
            
            model_class = GIMMVFI_F if "gimmvfi_f" in model_name else GIMMVFI_R
            model = model_class(**config["model"]["model_args"])
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            
            dtype = {"fp32": torch.float32, "fp16": torch.float16, "bfloat16": torch.bfloat16}.get(precision, torch.float32)
            
            model.to(dtype).eval()
            model.dtype = dtype

            if torch_compile:
                model = torch.compile(model)
                
            print(f"GIMM-VFI: Loaded {model_name} with intended precision {precision}")
            return (model,)
        except Exception as e:
            print(f"\033[31mGIMM-VFI Error: Failed to load model. Reason: {e}\033[0m")
            return (None,)

class GIMMVFI:
    # ... (The rest of the Fortress Edition code remains the same, no changes needed here)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gimmvfi_model": ("GIMMVFI_MODEL",),
                "images": ("IMAGE",),
                "interpolation_factor": ("INT", {"default": 2, "min": 2, "max": 16, "step": 1}),
                "safe_mode_chunk_size": ("INT", {"default": 16, "min": 2, "max": 64, "step": 1}),
                "ds_factor": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "output_flows": ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "flow_images")
    FUNCTION = "interpolate"
    CATEGORY = "GIMM-VFI"

    def interpolate(self, gimmvfi_model, images, interpolation_factor, safe_mode_chunk_size, ds_factor, seed, output_flows=False):
        if gimmvfi_model is None:
            return (images, torch.zeros(1, 64, 64, 3))
        if images is None or images.shape[0] < 2:
            return (images, torch.zeros(1, 64, 64, 3))
        try:
            effective_dtype, use_safe_mode, optimal_batch_size = preflight_check(gimmvfi_model, gimmvfi_model.dtype, images, interpolation_factor)
            if use_safe_mode:
                return self.interpolate_safe_mode(gimmvfi_model, images, interpolation_factor, safe_mode_chunk_size, ds_factor, seed, output_flows, effective_dtype)
            else:
                return self.interpolate_fast_mode(gimmvfi_model, images, interpolation_factor, optimal_batch_size, ds_factor, seed, output_flows, effective_dtype)
        except Exception as e:
            print(f"\033[31mGIMM-VFI Error: An unexpected error occurred: {e}\033[0m")
            import traceback
            traceback.print_exc()
            return (images, torch.zeros(1, 64, 64, 3))

    def _process_chunk(self, model, chunk_images_gpu, interpolation_factor, batch_size, ds_factor, effective_dtype, output_flows=False):
        device = chunk_images_gpu.device
        autocast_context = torch.autocast(device_type=device.type, dtype=effective_dtype) if effective_dtype != torch.float32 else nullcontext()
        num_input_frames = chunk_images_gpu.shape[0]
        if num_input_frames <= 1: return chunk_images_gpu, []
        total_output_frames = (num_input_frames - 1) * interpolation_factor + 1
        h, w = chunk_images_gpu.shape[2], chunk_images_gpu.shape[3]
        output_tensor_gpu = torch.empty(total_output_frames, 3, h, w, dtype=chunk_images_gpu.dtype, device=device)
        flows_gpu_list = []
        frame_pairs_I0 = [chunk_images_gpu[j] for j in range(num_input_frames - 1)]
        frame_pairs_I2 = [chunk_images_gpu[j+1] for j in range(num_input_frames - 1)]
        output_idx = 0
        pbar = comfy.utils.ProgressBar(len(frame_pairs_I0))
        with torch.no_grad(), autocast_context:
            for i in range(0, len(frame_pairs_I0), batch_size):
                batch_I0 = torch.stack(frame_pairs_I0[i:i+batch_size])
                batch_I2 = torch.stack(frame_pairs_I2[i:i+batch_size])
                padder = InputPadder(batch_I0.shape, 32)
                batch_I0_padded, batch_I2_padded = padder.pad(batch_I0, batch_I2)
                current_batch_size = batch_I0.shape[0]
                xs = torch.cat((batch_I0_padded.unsqueeze(2), batch_I2_padded.unsqueeze(2)), dim=2)
                timesteps_to_process = [t / interpolation_factor for t in range(interpolation_factor + 1)]
                coord_inputs = [(model.sample_coord_input(current_batch_size, xs.shape[-2:], [t], device=xs.device, upsample_ratio=ds_factor), None) for t in timesteps_to_process]
                timesteps_tensor = [t * torch.ones(current_batch_size).to(xs.device) for t in timesteps_to_process]
                all_outputs = model(xs, coord_inputs, t=timesteps_tensor, ds_factor=ds_factor)
                for batch_idx in range(current_batch_size):
                    if i + batch_idx == 0:
                        output_tensor_gpu[output_idx] = batch_I0[batch_idx]
                        output_idx +=1
                    for k in range(1, interpolation_factor):
                        frame_index_in_all_outputs = batch_idx * (interpolation_factor + 1) + k
                        unpadded_frame = padder.unpad(all_outputs["imgt_pred"][frame_index_in_all_outputs].unsqueeze(0))
                        output_tensor_gpu[output_idx] = unpadded_frame.detach().squeeze(0)
                        output_idx += 1
                    output_tensor_gpu[output_idx] = batch_I2[batch_idx]
                    if output_idx < total_output_frames -1 :
                         output_idx += 1
                pbar.update(current_batch_size)
        return output_tensor_gpu[:output_idx], flows_gpu_list

    def interpolate_fast_mode(self, model, images, interpolation_factor, batch_size, ds_factor, seed, output_flows, effective_dtype):
        torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        images_gpu = images.to(mm.get_torch_device(), non_blocking=True).permute(0, 3, 1, 2)
        output_tensor_gpu, _ = self._process_chunk(model, images_gpu, interpolation_factor, batch_size, ds_factor, effective_dtype, output_flows)
        image_tensors = output_tensor_gpu.permute(0, 2, 3, 1).cpu().float()
        flow_tensors = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
        return (image_tensors, flow_tensors)

    def interpolate_safe_mode(self, model, images, interpolation_factor, chunk_size, ds_factor, seed, output_flows, effective_dtype):
        final_images_cpu = []
        pbar = comfy.utils.ProgressBar(images.shape[0])
        for i in range(0, images.shape[0], chunk_size - 1):
            chunk = images[i:i + chunk_size]
            if chunk.shape[0] < 2: 
                if chunk.shape[0] > 0 and i > 0: final_images_cpu.append(chunk[0])
                continue
            print(f"\nProcessing chunk {i // (chunk_size - 1) + 1}/{int(np.ceil((images.shape[0])/(chunk_size-1)))} ({chunk.shape[0]} frames)...")
            torch.manual_seed(seed); torch.cuda.manual_seed(seed)
            chunk_gpu = chunk.to(mm.get_torch_device(), non_blocking=True).permute(0, 3, 1, 2)
            output_chunk_gpu, _ = self._process_chunk(model, chunk_gpu, interpolation_factor, 1, ds_factor, effective_dtype, output_flows)
            output_chunk_cpu = output_chunk_gpu.permute(0, 2, 3, 1).cpu().float()
            if i == 0:
                final_images_cpu.extend(list(torch.unbind(output_chunk_cpu)))
            else:
                final_images_cpu.extend(list(torch.unbind(output_chunk_cpu[1:])))
            pbar.update(chunk.shape[0] -1)
            mm.soft_empty_cache()
        if not final_images_cpu: return (images, torch.zeros(1, 64, 64, 3))
        image_tensors = torch.stack(final_images_cpu)
        flow_tensors = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
        return (image_tensors, flow_tensors)

NODE_CLASS_MAPPINGS = {
    "GIMMVFILoader": GIMMVFILoader,
    "GIMMVFI": GIMMVFI
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GIMMVFILoader": "GIMM-VFI Loader (Optimized)",
    "GIMMVFI": "GIMM-VFI Interpolate (Optimized)"
}
