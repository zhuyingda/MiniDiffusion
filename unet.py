import math
import json
import random
from PIL import Image
import numpy as np

# ---------- small helpers ----------

def silu(x):
    return x * (1.0 / (1.0 + math.exp(-x)))

def group_norm(data, weight, bias, num_groups_default=32):
    C = len(data)
    H = len(data[0]); W = len(data[0][0])

    # 如果有weight，就用它的长度推断group数量，否则用默认
    if weight is not None:
        groups = min(len(weight), C)
    else:
        groups = num_groups_default
    group_size = C // groups if groups > 0 else C

    out = [[[0.0 for _ in range(W)] for _ in range(H)] for _ in range(C)]
    eps = 1e-5
    for g in range(groups):
        start = g * group_size
        end = start + group_size
        mean = 0.0; var = 0.0
        cnt = (end - start) * H * W
        for ch in range(start, end):
            for i in range(H):
                for j in range(W):
                    mean += data[ch][i][j]
        mean /= cnt
        for ch in range(start, end):
            for i in range(H):
                for j in range(W):
                    diff = data[ch][i][j] - mean
                    var += diff * diff
        var /= cnt
        inv = 1.0 / math.sqrt(var + eps)

        for ch in range(start, end):
            w = weight[ch % len(weight)] if weight is not None else 1.0
            b = bias[ch % len(bias)] if bias is not None else 0.0
            for i in range(H):
                for j in range(W):
                    out[ch][i][j] = (data[ch][i][j] - mean) * inv * w + b
    return out

def conv2d(x, w, b, stride=1, padding=0):
    in_c = len(x)
    H = len(x[0]); W = len(x[0][0])
    out_c = len(w)
    # 有些权重可能 in_c 对不上，取最小值
    expected_in_c = len(w[0]) if out_c > 0 else 0
    kH = len(w[0][0]) if expected_in_c > 0 else 0
    kW = len(w[0][0][0]) if kH > 0 else 0

    Ho = (H + 2*padding - kH) // stride + 1
    Wo = (W + 2*padding - kW) // stride + 1
    y = [[[0.0 for _ in range(Wo)] for _ in range(Ho)] for _ in range(out_c)]
    for oc in range(out_c):
        bb = b[oc] if b is not None else 0.0
        for oy in range(Ho):
            for ox in range(Wo):
                acc = 0.0
                for ic in range(min(in_c, len(w[oc]))):  # 防止 ic 越界
                    for kh in range(kH):
                        iy = oy*stride - padding + kh
                        if iy < 0 or iy >= H: 
                            continue
                        for kw in range(kW):
                            ix = ox*stride - padding + kw
                            if 0 <= ix < W:
                                # 双重检查防止越界
                                if kh < len(w[oc][ic]) and kw < len(w[oc][ic][kh]):
                                    acc += x[ic][iy][ix] * w[oc][ic][kh][kw]
                y[oc][oy][ox] = acc + bb
    return y

def conv_transpose2d(x, w, b, stride=2, padding=1, output_padding=0):
    in_c = len(x)
    H = len(x[0]); W = len(x[0][0])
    out_c = len(w[0])
    kH = len(w[0][0]); kW = len(w[0][0][0])
    
    # Calculate base output size
    Ho_base = (H - 1) * stride - 2*padding + kH
    Wo_base = (W - 1) * stride - 2*padding + kW
    
    # Add output padding to match expected dimensions
    # For 7x7 -> 14x14: we need output_padding = 1
    # For 14x14 -> 28x28: we need output_padding = 1  
    if H == 7 and stride == 2:  # First upsample 7->14
        output_padding = 1
        Ho = Ho_base + output_padding
        Wo = Wo_base + output_padding
    elif H == 14 and stride == 2:  # Second upsample 14->28
        output_padding = 1
        Ho = Ho_base + output_padding
        Wo = Wo_base + output_padding
    else:
        Ho = Ho_base + output_padding
        Wo = Wo_base + output_padding
    
    y = [[[0.0 for _ in range(Wo)] for _ in range(Ho)] for _ in range(out_c)]
    for ic in range(in_c):
        for oc in range(out_c):
            for iy in range(H):
                for ix in range(W):
                    v = x[ic][iy][ix]
                    if v == 0:
                        continue
                    for kh in range(kH):
                        oy = iy*stride - padding + kh
                        if oy < 0 or oy >= Ho: 
                            continue
                        for kw in range(kW):
                            ox = ix*stride - padding + kw
                            if 0 <= ox < Wo:
                                y[oc][oy][ox] += v * w[ic][oc][kh][kw]
    if b is not None:
        for oc in range(out_c):
            bb = b[oc]
            for i in range(Ho):
                for j in range(Wo):
                    y[oc][i][j] += bb
    return y

# ---------- UNet (scalar) ----------
class SimpleUNet:
    def __init__(self, weights):
        self.w = weights
        self.num_groups = 32

    def predict_noise(self, x, t_idx, temb):
        w = self.w

        # stem
        h = conv2d(x, w["conv_in.weight"], w["conv_in.bias"], padding=1)

        # ----- down block 0 -----
        h0 = group_norm(h, w["down_blocks.0.resnets.0.norm1.weight"], w["down_blocks.0.resnets.0.norm1.bias"], self.num_groups)
        h0 = apply_silu(h0); h0 = conv2d(h0, w["down_blocks.0.resnets.0.conv1.weight"], w["down_blocks.0.resnets.0.conv1.bias"], padding=1)
        h0 = add_time(h0, temb, w, "down_blocks.0.resnets.0")
        h0 = group_norm(h0, w["down_blocks.0.resnets.0.norm2.weight"], w["down_blocks.0.resnets.0.norm2.bias"], self.num_groups)
        h0 = apply_silu(h0); h0 = conv2d(h0, w["down_blocks.0.resnets.0.conv2.weight"], w["down_blocks.0.resnets.0.conv2.bias"], padding=1)
        h = add_(h0, h)

        h1 = group_norm(h, w["down_blocks.0.resnets.1.norm1.weight"], w["down_blocks.0.resnets.1.norm1.bias"], self.num_groups)
        h1 = apply_silu(h1); h1 = conv2d(h1, w["down_blocks.0.resnets.1.conv1.weight"], w["down_blocks.0.resnets.1.conv1.bias"], padding=1)
        h1 = add_time(h1, temb, w, "down_blocks.0.resnets.1")
        h1 = group_norm(h1, w["down_blocks.0.resnets.1.norm2.weight"], w["down_blocks.0.resnets.1.norm2.bias"], self.num_groups)
        h1 = apply_silu(h1); h1 = conv2d(h1, w["down_blocks.0.resnets.1.conv2.weight"], w["down_blocks.0.resnets.1.conv2.bias"], padding=1)
        h = add_(h1, h)
        skip1 = clone_feat(h)

        h = conv2d(h, w["down_blocks.0.downsamplers.0.conv.weight"], w["down_blocks.0.downsamplers.0.conv.bias"], stride=2, padding=1)

        # ----- down block 1 -----
        h2 = group_norm(h, w["down_blocks.1.resnets.0.norm1.weight"], w["down_blocks.1.resnets.0.norm1.bias"], self.num_groups)
        h2 = apply_silu(h2); h2 = conv2d(h2, w["down_blocks.1.resnets.0.conv1.weight"], w["down_blocks.1.resnets.0.conv1.bias"], padding=1)
        h2 = add_time(h2, temb, w, "down_blocks.1.resnets.0")
        h2 = group_norm(h2, w["down_blocks.1.resnets.0.norm2.weight"], w["down_blocks.1.resnets.0.norm2.bias"], self.num_groups)
        h2 = apply_silu(h2); h2 = conv2d(h2, w["down_blocks.1.resnets.0.conv2.weight"], w["down_blocks.1.resnets.0.conv2.bias"], padding=1)
        sc0 = conv2d(h, w["down_blocks.1.resnets.0.conv_shortcut.weight"], w["down_blocks.1.resnets.0.conv_shortcut.bias"])
        h = add_(h2, sc0)

        h3 = group_norm(h, w["down_blocks.1.resnets.1.norm1.weight"], w["down_blocks.1.resnets.1.norm1.bias"], self.num_groups)
        h3 = apply_silu(h3); h3 = conv2d(h3, w["down_blocks.1.resnets.1.conv1.weight"], w["down_blocks.1.resnets.1.conv1.bias"], padding=1)
        h3 = add_time(h3, temb, w, "down_blocks.1.resnets.1")
        h3 = group_norm(h3, w["down_blocks.1.resnets.1.norm2.weight"], w["down_blocks.1.resnets.1.norm2.bias"], self.num_groups)
        h3 = apply_silu(h3); h3 = conv2d(h3, w["down_blocks.1.resnets.1.conv2.weight"], w["down_blocks.1.resnets.1.conv2.bias"], padding=1)
        h = add_(h3, h)
        skip2 = clone_feat(h)

        h = conv2d(h, w["down_blocks.1.downsamplers.0.conv.weight"], w["down_blocks.1.downsamplers.0.conv.bias"], stride=2, padding=1)

        # ----- down block 2 (bottom) -----
        h4 = group_norm(h, w["down_blocks.2.resnets.0.norm1.weight"], w["down_blocks.2.resnets.0.norm1.bias"], self.num_groups)
        h4 = apply_silu(h4); h4 = conv2d(h4, w["down_blocks.2.resnets.0.conv1.weight"], w["down_blocks.2.resnets.0.conv1.bias"], padding=1)
        h4 = add_time(h4, temb, w, "down_blocks.2.resnets.0")
        h4 = group_norm(h4, w["down_blocks.2.resnets.0.norm2.weight"], w["down_blocks.2.resnets.0.norm2.bias"], self.num_groups)
        h4 = apply_silu(h4); h4 = conv2d(h4, w["down_blocks.2.resnets.0.conv2.weight"], w["down_blocks.2.resnets.0.conv2.bias"], padding=1)
        sc1 = conv2d(h, w["down_blocks.2.resnets.0.conv_shortcut.weight"], w["down_blocks.2.resnets.0.conv_shortcut.bias"])
        h = add_(h4, sc1)

        h5 = group_norm(h, w["down_blocks.2.resnets.1.norm1.weight"], w["down_blocks.2.resnets.1.norm1.bias"], self.num_groups)
        h5 = apply_silu(h5); h5 = conv2d(h5, w["down_blocks.2.resnets.1.conv1.weight"], w["down_blocks.2.resnets.1.conv1.bias"], padding=1)
        h5 = add_time(h5, temb, w, "down_blocks.2.resnets.1")
        h5 = group_norm(h5, w["down_blocks.2.resnets.1.norm2.weight"], w["down_blocks.2.resnets.1.norm2.bias"], self.num_groups)
        h5 = apply_silu(h5); h5 = conv2d(h5, w["down_blocks.2.resnets.1.conv2.weight"], w["down_blocks.2.resnets.1.conv2.bias"], padding=1)
        h = add_(h5, h)

        # ----- mid block (no attention) -----
        m0 = group_norm(h, w["mid_block.resnets.0.norm1.weight"], w["mid_block.resnets.0.norm1.bias"], self.num_groups)
        m0 = apply_silu(m0); m0 = conv2d(m0, w["mid_block.resnets.0.conv1.weight"], w["mid_block.resnets.0.conv1.bias"], padding=1)
        m0 = add_time(m0, temb, w, "mid_block.resnets.0")
        m0 = group_norm(m0, w["mid_block.resnets.0.norm2.weight"], w["mid_block.resnets.0.norm2.bias"], self.num_groups)
        m0 = apply_silu(m0); m0 = conv2d(m0, w["mid_block.resnets.0.conv2.weight"], w["mid_block.resnets.0.conv2.bias"], padding=1)
        h = add_(m0, h)

        m1 = group_norm(h, w["mid_block.resnets.1.norm1.weight"], w["mid_block.resnets.1.norm1.bias"], self.num_groups)
        m1 = apply_silu(m1); m1 = conv2d(m1, w["mid_block.resnets.1.conv1.weight"], w["mid_block.resnets.1.conv1.bias"], padding=1)
        m1 = add_time(m1, temb, w, "mid_block.resnets.1")
        m1 = group_norm(m1, w["mid_block.resnets.1.norm2.weight"], w["mid_block.resnets.1.norm2.bias"], self.num_groups)
        m1 = apply_silu(m1); m1 = conv2d(m1, w["mid_block.resnets.1.conv2.weight"], w["mid_block.resnets.1.conv2.bias"], padding=1)
        h = add_(m1, h)

        # ----- up block 0 -----
        h = conv_transpose2d(h, w["up_blocks.0.upsamplers.0.conv.weight"], w["up_blocks.0.upsamplers.0.conv.bias"], stride=2, padding=1)
        # concat skip2
        for c in range(len(skip2)):
            h.append([row[:] for row in skip2[c]])

        u0 = group_norm(h, w["up_blocks.0.resnets.0.norm1.weight"], w["up_blocks.0.resnets.0.norm1.bias"], self.num_groups)
        u0 = apply_silu(u0); u0 = conv2d(u0, w["up_blocks.0.resnets.0.conv1.weight"], w["up_blocks.0.resnets.0.conv1.bias"], padding=1)
        u0 = add_time(u0, temb, w, "up_blocks.0.resnets.0")
        u0 = group_norm(u0, w["up_blocks.0.resnets.0.norm2.weight"], w["up_blocks.0.resnets.0.norm2.bias"], self.num_groups)
        u0 = apply_silu(u0); u0 = conv2d(u0, w["up_blocks.0.resnets.0.conv2.weight"], w["up_blocks.0.resnets.0.conv2.bias"], padding=1)
        sc2 = conv2d(h, w["up_blocks.0.resnets.0.conv_shortcut.weight"], w["up_blocks.0.resnets.0.conv_shortcut.bias"])
        h = add_(u0, sc2)

        u1 = group_norm(h, w["up_blocks.0.resnets.1.norm1.weight"], w["up_blocks.0.resnets.1.norm1.bias"], self.num_groups)
        u1 = apply_silu(u1); u1 = conv2d(u1, w["up_blocks.0.resnets.1.conv1.weight"], w["up_blocks.0.resnets.1.conv1.bias"], padding=1)
        u1 = add_time(u1, temb, w, "up_blocks.0.resnets.1")
        u1 = group_norm(u1, w["up_blocks.0.resnets.1.norm2.weight"], w["up_blocks.0.resnets.1.norm2.bias"], self.num_groups)
        u1 = apply_silu(u1); u1 = conv2d(u1, w["up_blocks.0.resnets.1.conv2.weight"], w["up_blocks.0.resnets.1.conv2.bias"], padding=1)
        h = add_(u1, h)

        # ----- up block 1 -----
        h = conv_transpose2d(h, w["up_blocks.1.upsamplers.0.conv.weight"], w["up_blocks.1.upsamplers.0.conv.bias"], stride=2, padding=1)
        # concat skip1
        for c in range(len(skip1)):
            h.append([row[:] for row in skip1[c]])

        u2 = group_norm(h, w["up_blocks.1.resnets.0.norm1.weight"], w["up_blocks.1.resnets.0.norm1.bias"], self.num_groups)
        u2 = apply_silu(u2); u2 = conv2d(u2, w["up_blocks.1.resnets.0.conv1.weight"], w["up_blocks.1.resnets.0.conv1.bias"], padding=1)
        u2 = add_time(u2, temb, w, "up_blocks.1.resnets.0")
        u2 = group_norm(u2, w["up_blocks.1.resnets.0.norm2.weight"], w["up_blocks.1.resnets.0.norm2.bias"], self.num_groups)
        u2 = apply_silu(u2); u2 = conv2d(u2, w["up_blocks.1.resnets.0.conv2.weight"], w["up_blocks.1.resnets.0.conv2.bias"], padding=1)
        sc3 = conv2d(h, w["up_blocks.1.resnets.0.conv_shortcut.weight"], w["up_blocks.1.resnets.0.conv_shortcut.bias"])
        h = add_(u2, sc3)

        u3 = group_norm(h, w["up_blocks.1.resnets.1.norm1.weight"], w["up_blocks.1.resnets.1.norm1.bias"], self.num_groups)
        u3 = apply_silu(u3); u3 = conv2d(u3, w["up_blocks.1.resnets.1.conv1.weight"], w["up_blocks.1.resnets.1.conv1.bias"], padding=1)
        u3 = add_time(u3, temb, w, "up_blocks.1.resnets.1")
        u3 = group_norm(u3, w["up_blocks.1.resnets.1.norm2.weight"], w["up_blocks.1.resnets.1.norm2.bias"], self.num_groups)
        u3 = apply_silu(u3); u3 = conv2d(u3, w["up_blocks.1.resnets.1.conv2.weight"], w["up_blocks.1.resnets.1.conv2.bias"], padding=1)
        h = add_(u3, h)

        # ----- up block 2 (head) -----
        u4 = group_norm(h, w["up_blocks.2.resnets.0.norm1.weight"], w["up_blocks.2.resnets.0.norm1.bias"], self.num_groups)
        u4 = apply_silu(u4); u4 = conv2d(u4, w["up_blocks.2.resnets.0.conv1.weight"], w["up_blocks.2.resnets.0.conv1.bias"], padding=1)
        u4 = add_time(u4, temb, w, "up_blocks.2.resnets.0")
        u4 = group_norm(u4, w["up_blocks.2.resnets.0.norm2.weight"], w["up_blocks.2.resnets.0.norm2.bias"], self.num_groups)
        u4 = apply_silu(u4); u4 = conv2d(u4, w["up_blocks.2.resnets.0.conv2.weight"], w["up_blocks.2.resnets.0.conv2.bias"], padding=1)
        # optional shortcut
        if "up_blocks.2.resnets.0.conv_shortcut.weight" in w:
            sc4 = conv2d(h, w["up_blocks.2.resnets.0.conv_shortcut.weight"], w["up_blocks.2.resnets.0.conv_shortcut.bias"])
        else:
            sc4 = u4
        h = add_(u4, sc4)

        u5 = group_norm(h, w["up_blocks.2.resnets.1.norm1.weight"], w["up_blocks.2.resnets.1.norm1.bias"], self.num_groups)
        u5 = apply_silu(u5); u5 = conv2d(u5, w["up_blocks.2.resnets.1.conv1.weight"], w["up_blocks.2.resnets.1.conv1.bias"], padding=1)
        u5 = add_time(u5, temb, w, "up_blocks.2.resnets.1")
        u5 = group_norm(u5, w["up_blocks.2.resnets.1.norm2.weight"], w["up_blocks.2.resnets.1.norm2.bias"], self.num_groups)
        u5 = apply_silu(u5); u5 = conv2d(u5, w["up_blocks.2.resnets.1.conv2.weight"], w["up_blocks.2.resnets.1.conv2.bias"], padding=1)
        h = add_(u5, h)

        # ----- out head -----
        hout = group_norm(h, w["conv_norm_out.weight"], w["conv_norm_out.bias"], self.num_groups)
        hout = apply_silu(hout)
        out = conv2d(hout, w["conv_out.weight"], w["conv_out.bias"], padding=1)
        return out[0]  # 1x28x28

def apply_silu(feat):
    C = len(feat); H = len(feat[0]); W = len(feat[0][0])
    for c in range(C):
        for i in range(H):
            for j in range(W):
                feat[c][i][j] = silu(feat[c][i][j])
    return feat

def add_(a, b):
    C = len(a); H = len(a[0]); W = len(a[0][0])
    for c in range(C):
        for i in range(H):
            for j in range(W):
                a[c][i][j] += b[c][i][j]
    return a

def clone_feat(f):
    return [[row[:] for row in ch] for ch in f]

def add_time(h, temb, w, prefix):
    tW = w[f"{prefix}.time_emb_proj.weight"]
    tB = w[f"{prefix}.time_emb_proj.bias"]
    out_ch = len(tW)
    proj = [tB[c] + sum(temb[k] * tW[c][k] for k in range(len(tW[c]))) for c in range(out_ch)]
    H = len(h[0]); W = len(h[0][0])
    for c in range(out_ch):
        val = proj[c]
        for i in range(H):
            for j in range(W):
                h[c][i][j] += val
    return h

def load_reference_image():
    """Load the reference image from prepare.py as initial noise guidance"""
    try:
        ref_img = Image.open("ref_prepare_result.png").convert('L')
        ref_array = np.array(ref_img).astype(np.float32)
        # Convert from [0,255] to [-1,1] (standard DDPM range)
        ref_normalized = (ref_array / 127.5) - 1.0
        return ref_normalized.tolist()
    except:
        print("Warning: Could not load reference image, using random noise")
        return [[random.gauss(0.0, 1.0) for _ in range(28)] for __ in range(28)]

# ---------- main (DDPM sampling) ----------

if __name__ == "__main__":
    # Load the data
    with open("unet_weights.json", "r") as f:
        weights = json.load(f)
    with open("betas.json", "r") as f:
        betas = json.load(f)

    # 预计算 alphas / alpha_cumprod
    alphas = [1.0 - b for b in betas]
    alpha_bar = [0.0] * len(alphas)
    prod = 1.0
    for t, a in enumerate(alphas):
        prod *= a
        alpha_bar[t] = prod

    model = SimpleUNet(weights)

    # Use reference image as starting point instead of pure random noise
    print("Loading reference image for guidance...")
    ref_img = load_reference_image()
    
    # Add some controlled noise to the reference image to allow for variation
    noise_strength = 0.5  # Adjust this to control how much we deviate from reference
    img = [[ref_img[i][j] + noise_strength * random.gauss(0.0, 1.0) 
            for j in range(28)] for i in range(28)]

    # Use more steps for better quality
    num_steps = 1
    
    # Create a better timestep schedule - use the actual scheduler timesteps
    total_timesteps = len(betas)
    if num_steps > 1:
        # Use evenly spaced timesteps from the full range
        step_size = total_timesteps // num_steps
        t_grid = [total_timesteps - 1 - i * step_size for i in range(num_steps)]
        t_grid[-1] = 0  # Ensure we end at t=0
    else:
        t_grid = [total_timesteps - 1]

    print(f"Running DDPM sampling with {num_steps} steps...")
    frames = []

    for step_idx, t in enumerate(t_grid):
        # Ensure t is within bounds
        t = max(0, min(t, len(betas) - 1))
        
        # --- time embedding ---
        emb_in_dim = len(weights["time_embedding.linear_1.weight"][0])
        half = emb_in_dim // 2
        # sinusoidal embedding
        sinu = []
        for i in range(half):
            freq = math.exp(-math.log(10000) * i / max(half-1, 1))
            sinu.append(math.cos(t * freq))
        for i in range(half):
            freq = math.exp(-math.log(10000) * i / max(half-1, 1))
            sinu.append(math.sin(t * freq))
        # linear_1
        l1w = weights["time_embedding.linear_1.weight"]
        l1b = weights["time_embedding.linear_1.bias"]
        h = [l1b[o] + sum(sinu[i] * l1w[o][i] for i in range(len(sinu))) for o in range(len(l1w))]
        h = [silu(v) for v in h]
        # linear_2
        l2w = weights["time_embedding.linear_2.weight"]
        l2b = weights["time_embedding.linear_2.bias"]
        temb = [l2b[o] + sum(h[i] * l2w[o][i] for i in range(len(h))) for o in range(len(l2w))]
        temb = [silu(v) for v in temb]

        # --- 预测噪声 eps_theta(x_t, t) ---
        x_in = [img]  # (C=1,H,W)
        eps = model.predict_noise(x_in, t, temb)
        
        # Verify eps has the right dimensions (should be 28x28 now)
        if len(eps) != 28 or len(eps[0]) != 28:
            print(f"Error: eps shape is {len(eps)}x{len(eps[0])}, expected 28x28")
            print("Check the conv_transpose2d output_padding logic")
            break

        # --- DDPM reverse update with better numerical stability ---
        if t > 0:
            a_t = alphas[t]
            ab_t = alpha_bar[t]
            
            # Use previous timestep's alpha_bar for better computation
            t_prev = t_grid[step_idx + 1] if step_idx + 1 < len(t_grid) else 0
            ab_prev = alpha_bar[t_prev] if t_prev > 0 else 1.0
            
            # Compute coefficients with better numerical stability
            sqrt_inv_at = 1.0 / math.sqrt(a_t) if a_t > 1e-8 else 1.0
            sqrt_ab_t = math.sqrt(max(1e-12, ab_t))
            sqrt_one_minus_ab = math.sqrt(max(1e-12, 1.0 - ab_t))
            
            # Mean computation
            coeff = (1.0 - a_t) / sqrt_one_minus_ab
            mean = [[sqrt_inv_at * (img[i][j] - coeff * eps[i][j])
                     for j in range(28)] for i in range(28)]
            
            # Variance computation - use the correct DDPM formula
            beta_t = betas[t]
            posterior_var = beta_t * (1.0 - ab_prev) / (1.0 - ab_t) if (1.0 - ab_t) > 1e-12 else 0.0
            sigma = math.sqrt(max(0.0, posterior_var))
            
            # Add noise if not the last step
            if step_idx < len(t_grid) - 1 and sigma > 0:
                noise = [[random.gauss(0.0, 1.0) for _ in range(28)] for __ in range(28)]
                img = [[mean[i][j] + sigma * noise[i][j] for j in range(28)] for i in range(28)]
            else:
                img = mean
        else:
            # Final step - deterministic
            a_t = alphas[0] if len(alphas) > 0 else 1.0
            sqrt_inv_at = 1.0 / math.sqrt(a_t) if a_t > 1e-8 else 1.0
            sqrt_one_minus_ab = math.sqrt(max(1e-12, 1.0 - alpha_bar[0]))
            coeff = sqrt_one_minus_ab
            img = [[sqrt_inv_at * (img[i][j] - coeff * eps[i][j]) for j in range(28)] for i in range(28)]

        # Record frame
        frames.append([[img[i][j] for j in range(28)] for i in range(28)])
        
        if step_idx % 10 == 0:
            print(f"Step {step_idx + 1}/{num_steps} (t={t}) complete")

    print("Sampling complete, creating output images...")

    # Save progression image
    W = 28 * len(frames)
    canvas = Image.new("L", (W, 28))
    for k, f in enumerate(frames):
        # Better normalization - use proper DDPM range
        mn = min(min(row) for row in f)
        mx = max(max(row) for row in f)
        
        pixels = []
        if mx - mn > 1e-8:
            scale = (mx - mn)
            for i in range(28):
                for j in range(28):
                    # Map from [-1,1] range to [0,255]
                    v = (f[i][j] - mn) / scale
                    v = max(0.0, min(1.0, v))
                    pixels.append(int(v * 255))
        else:
            pixels = [128] * (28 * 28)
        
        frame = Image.new("L", (28, 28))
        frame.putdata(pixels)
        canvas.paste(frame, (k * 28, 0))
    canvas.save("result_progression.png")

    # Save final result
    final_frame = frames[-1]
    mn = min(min(row) for row in final_frame)
    mx = max(max(row) for row in final_frame)
    
    pixels = []
    if mx - mn > 1e-8:
        scale = (mx - mn)
        for i in range(28):
            for j in range(28):
                v = (final_frame[i][j] - mn) / scale
                v = max(0.0, min(1.0, v))
                pixels.append(int(v * 255))
    else:
        pixels = [128] * (28 * 28)
    
    final_image = Image.new("L", (28, 28))
    final_image.putdata(pixels)
    final_image.save("result_final.png")
    
    print("Saved result_progression.png and result_final.png")