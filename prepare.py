import math, json, os
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline
from PIL import Image

# 配置
MODEL_ID = "1aurent/ddpm-mnist"   # 28x28 灰度、已训练 DDPM
NUM_INFERENCE_STEPS = 50          # 增加采样步数以获得更好的结果
SEED = 42                         # 改变种子试试不同结果
OUT_JSON = "scalar_pack.json"     # 给 scalar.py 使用的"打包数据"
REF_PNG = "ref_prepare_result.png"  # 仅供参考：prepare端也存一张结果

def to_pylist(x: torch.Tensor):
    return x.detach().cpu().tolist()

def main():
    torch.manual_seed(SEED)

    # 使用正确的 DDPMPipeline
    pipeline = DDPMPipeline.from_pretrained(MODEL_ID)
    
    # 生成图像
    print(f"Generating image with {NUM_INFERENCE_STEPS} steps...")
    image = pipeline(num_inference_steps=NUM_INFERENCE_STEPS, generator=torch.Generator().manual_seed(SEED)).images[0]
    
    # 保存结果
    image.save(REF_PNG)
    print(f"Generated image saved to {REF_PNG}")
    
    # 为了保持与原始脚本的兼容性，我们也生成一些数据
    # 但这次使用正确的pipeline组件
    unet = pipeline.unet
    scheduler = pipeline.scheduler
    
    # 设置调度器步数
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    
    # 创建数据包
    H = W = unet.config.sample_size  # 28
    x_T = torch.randn(1, 1, H, W, dtype=torch.float32)
    
    alphas_cumprod = scheduler.alphas_cumprod.cpu().numpy().tolist()
    pack = {
        "seed": SEED,
        "H": H, "W": W,
        "timesteps": [int(t.item()) for t in scheduler.timesteps],
        "alphas_cumprod": alphas_cumprod,
        "x_T": to_pylist(x_T[0,0]),   # 存HW二维
        "steps": [],                  # 每一步的 {t, eps, z, coeffs...}
        "conv_in": {},                # 卷积权重/偏置 + 小测例
    }

    # 存 conv_in 权重与一个小测例（8x8）
    w = unet.conv_in.weight.detach().cpu()  # (Cout, Cin, 3, 3)
    b = unet.conv_in.bias.detach().cpu() if unet.conv_in.bias is not None else torch.zeros(w.shape[0])
    x_small = torch.randn(1,1,8,8)
    y_ref = F.conv2d(x_small, w, b, stride=1, padding=1)
    pack["conv_in"] = {
        "weight": to_pylist(w),           # [Cout][Cin][KH][KW]
        "bias": to_pylist(b),             # [Cout]
        "x_small": to_pylist(x_small[0,0]), # [8][8]
        "y_ref": to_pylist(y_ref[0]),        # [Cout][8][8]
        "stride": [1,1],
        "padding": [1,1],
        "kernel": [3,3],
    }

    # 使用正确的调度器进行采样（用于记录数据）
    x = x_T.clone()
    
    for idx, t in enumerate(scheduler.timesteps):
        # 使用调度器的scale_model_input方法
        model_input = scheduler.scale_model_input(x, t)
        
        # 预测噪声
        with torch.no_grad():
            noise_pred = unet(model_input, t).sample
        
        # 使用调度器的step方法进行正确的采样
        scheduler_output = scheduler.step(noise_pred, t, x)
        x = scheduler_output.prev_sample
        
        # 记录这一步的数据（简化版本）
        pack["steps"].append({
            "t": int(t.item()),
            "eps": to_pylist(noise_pred[0,0]),
            "z": to_pylist(torch.randn_like(x)[0,0]) if idx < len(scheduler.timesteps) - 1 else to_pylist(torch.zeros_like(x)[0,0]),
            "sqrt_recip_alpha_t": 1.0,  # 简化
            "coeff": 1.0,               # 简化
            "sigma_t": 0.0              # 简化
        })

    # 写 JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(pack, f)
    print(f"Saved pack -> {OUT_JSON}")

if __name__ == "__main__":
    main()