const fs = require('fs');
const path = require('path');
const { PNG } = require('pngjs');

// ---------- Small Helpers ----------

function silu(x) {
    return x * (1.0 / (1.0 + Math.exp(-x)));
}

function groupNorm(data, weight, bias, numGroupsDefault = 32) {
    const C = data.length;
    const H = data[0].length;
    const W = data[0][0].length;

    // If weight exists, use its length to infer group count, otherwise use default
    let groups;
    if (weight !== null && weight !== undefined) {
        groups = Math.min(weight.length, C);
    } else {
        groups = numGroupsDefault;
    }
    const groupSize = groups > 0 ? Math.floor(C / groups) : C;

    const out = Array(C).fill().map(() => 
        Array(H).fill().map(() => Array(W).fill(0.0))
    );
    
    const eps = 1e-5;
    
    for (let g = 0; g < groups; g++) {
        const start = g * groupSize;
        const end = start + groupSize;
        let mean = 0.0;
        let variance = 0.0;
        const cnt = (end - start) * H * W;
        
        // Calculate mean
        for (let ch = start; ch < end; ch++) {
            for (let i = 0; i < H; i++) {
                for (let j = 0; j < W; j++) {
                    mean += data[ch][i][j];
                }
            }
        }
        mean /= cnt;
        
        // Calculate variance
        for (let ch = start; ch < end; ch++) {
            for (let i = 0; i < H; i++) {
                for (let j = 0; j < W; j++) {
                    const diff = data[ch][i][j] - mean;
                    variance += diff * diff;
                }
            }
        }
        variance /= cnt;
        const inv = 1.0 / Math.sqrt(variance + eps);
        
        // Apply normalization
        for (let ch = start; ch < end; ch++) {
            const w = (weight !== null && weight !== undefined) ? weight[ch % weight.length] : 1.0;
            const b = (bias !== null && bias !== undefined) ? bias[ch % bias.length] : 0.0;
            for (let i = 0; i < H; i++) {
                for (let j = 0; j < W; j++) {
                    out[ch][i][j] = (data[ch][i][j] - mean) * inv * w + b;
                }
            }
        }
    }
    
    return out;
}

function conv2d(x, w, b, stride = 1, padding = 0) {
    const inC = x.length;
    const H = x[0].length;
    const W = x[0][0].length;
    const outC = w.length;
    
    // Some weights might not match input channels, take minimum
    const expectedInC = outC > 0 ? w[0].length : 0;
    const kH = expectedInC > 0 ? w[0][0].length : 0;
    const kW = kH > 0 ? w[0][0][0].length : 0;
    
    const Ho = Math.floor((H + 2 * padding - kH) / stride) + 1;
    const Wo = Math.floor((W + 2 * padding - kW) / stride) + 1;
    
    // Debug: check for invalid dimensions
    if (Ho <= 0 || Wo <= 0 || !isFinite(Ho) || !isFinite(Wo)) {
        console.error(`Invalid output dimensions: Ho=${Ho}, Wo=${Wo}, H=${H}, W=${W}, kH=${kH}, kW=${kW}, stride=${stride}, padding=${padding}`);
        return [[[0]]]; // Return minimal valid result
    }
    
    const y = Array(outC).fill().map(() => 
        Array(Ho).fill().map(() => Array(Wo).fill(0.0))
    );
    
    for (let oc = 0; oc < outC; oc++) {
        const bb = (b !== null && b !== undefined) ? b[oc] : 0.0;
        for (let oy = 0; oy < Ho; oy++) {
            for (let ox = 0; ox < Wo; ox++) {
                let acc = 0.0;
                for (let ic = 0; ic < Math.min(inC, w[oc].length); ic++) {
                    for (let kh = 0; kh < kH; kh++) {
                        const iy = oy * stride - padding + kh;
                        if (iy < 0 || iy >= H) continue;
                        
                        for (let kw = 0; kw < kW; kw++) {
                            const ix = ox * stride - padding + kw;
                            if (ix >= 0 && ix < W) {
                                // Double check to prevent out of bounds
                                if (kh < w[oc][ic].length && kw < w[oc][ic][kh].length) {
                                    acc += x[ic][iy][ix] * w[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                }
                y[oc][oy][ox] = acc + bb;
            }
        }
    }
    
    return y;
}

function convTranspose2d(x, w, b, stride = 2, padding = 1, outputPadding = 0) {
    const inC = x.length;
    const H = x[0].length;
    const W = x[0][0].length;
    const outC = w[0].length;
    const kH = w[0][0].length;
    const kW = w[0][0][0].length;
    
    // Calculate base output size
    const HoBase = (H - 1) * stride - 2 * padding + kH;
    const WoBase = (W - 1) * stride - 2 * padding + kW;
    
    // Add output padding to match expected dimensions
    let Ho, Wo;
    if (H === 7 && stride === 2) { // First upsample 7->14
        outputPadding = 1;
        Ho = HoBase + outputPadding;
        Wo = WoBase + outputPadding;
    } else if (H === 14 && stride === 2) { // Second upsample 14->28
        outputPadding = 1;
        Ho = HoBase + outputPadding;
        Wo = WoBase + outputPadding;
    } else {
        Ho = HoBase + outputPadding;
        Wo = WoBase + outputPadding;
    }
    
    // Debug: check for invalid dimensions
    if (Ho <= 0 || Wo <= 0 || !isFinite(Ho) || !isFinite(Wo)) {
        console.error(`Invalid transpose output dimensions: Ho=${Ho}, Wo=${Wo}, H=${H}, W=${W}, kH=${kH}, kW=${kW}, stride=${stride}, padding=${padding}`);
        return [[[0]]]; // Return minimal valid result
    }
    
    const y = Array(outC).fill().map(() => 
        Array(Ho).fill().map(() => Array(Wo).fill(0.0))
    );
    
    for (let ic = 0; ic < inC; ic++) {
        for (let oc = 0; oc < outC; oc++) {
            for (let iy = 0; iy < H; iy++) {
                for (let ix = 0; ix < W; ix++) {
                    const v = x[ic][iy][ix];
                    if (v === 0) continue;
                    
                    for (let kh = 0; kh < kH; kh++) {
                        const oy = iy * stride - padding + kh;
                        if (oy < 0 || oy >= Ho) continue;
                        
                        for (let kw = 0; kw < kW; kw++) {
                            const ox = ix * stride - padding + kw;
                            if (ox >= 0 && ox < Wo) {
                                y[oc][oy][ox] += v * w[ic][oc][kh][kw];
                            }
                        }
                    }
                }
            }
        }
    }
    
    if (b !== null && b !== undefined) {
        for (let oc = 0; oc < outC; oc++) {
            const bb = b[oc];
            for (let i = 0; i < Ho; i++) {
                for (let j = 0; j < Wo; j++) {
                    y[oc][i][j] += bb;
                }
            }
        }
    }
    
    return y;
}

// ---------- UNet (JavaScript) ----------

class SimpleUNet {
    constructor(weights) {
        this.w = weights;
        this.numGroups = 32;
    }

    predictNoise(x, tIdx, temb) {
        const w = this.w;

        // stem
        let h = conv2d(x, w["conv_in.weight"], w["conv_in.bias"], 1, 1);

        // ----- down block 0 -----
        let h0 = groupNorm(h, w["down_blocks.0.resnets.0.norm1.weight"], w["down_blocks.0.resnets.0.norm1.bias"], this.numGroups);
        h0 = applySilu(h0);
        h0 = conv2d(h0, w["down_blocks.0.resnets.0.conv1.weight"], w["down_blocks.0.resnets.0.conv1.bias"], 1, 1);
        h0 = addTime(h0, temb, w, "down_blocks.0.resnets.0");
        h0 = groupNorm(h0, w["down_blocks.0.resnets.0.norm2.weight"], w["down_blocks.0.resnets.0.norm2.bias"], this.numGroups);
        h0 = applySilu(h0);
        h0 = conv2d(h0, w["down_blocks.0.resnets.0.conv2.weight"], w["down_blocks.0.resnets.0.conv2.bias"], 1, 1);
        h = add(h0, h);

        let h1 = groupNorm(h, w["down_blocks.0.resnets.1.norm1.weight"], w["down_blocks.0.resnets.1.norm1.bias"], this.numGroups);
        h1 = applySilu(h1);
        h1 = conv2d(h1, w["down_blocks.0.resnets.1.conv1.weight"], w["down_blocks.0.resnets.1.conv1.bias"], 1, 1);
        h1 = addTime(h1, temb, w, "down_blocks.0.resnets.1");
        h1 = groupNorm(h1, w["down_blocks.0.resnets.1.norm2.weight"], w["down_blocks.0.resnets.1.norm2.bias"], this.numGroups);
        h1 = applySilu(h1);
        h1 = conv2d(h1, w["down_blocks.0.resnets.1.conv2.weight"], w["down_blocks.0.resnets.1.conv2.bias"], 1, 1);
        h = add(h1, h);
        const skip1 = cloneFeat(h);

        h = conv2d(h, w["down_blocks.0.downsamplers.0.conv.weight"], w["down_blocks.0.downsamplers.0.conv.bias"], 2, 1);

        // ----- down block 1 -----
        let h2 = groupNorm(h, w["down_blocks.1.resnets.0.norm1.weight"], w["down_blocks.1.resnets.0.norm1.bias"], this.numGroups);
        h2 = applySilu(h2);
        h2 = conv2d(h2, w["down_blocks.1.resnets.0.conv1.weight"], w["down_blocks.1.resnets.0.conv1.bias"], 1, 1);
        h2 = addTime(h2, temb, w, "down_blocks.1.resnets.0");
        h2 = groupNorm(h2, w["down_blocks.1.resnets.0.norm2.weight"], w["down_blocks.1.resnets.0.norm2.bias"], this.numGroups);
        h2 = applySilu(h2);
        h2 = conv2d(h2, w["down_blocks.1.resnets.0.conv2.weight"], w["down_blocks.1.resnets.0.conv2.bias"], 1, 1);
        const sc0 = conv2d(h, w["down_blocks.1.resnets.0.conv_shortcut.weight"], w["down_blocks.1.resnets.0.conv_shortcut.bias"]);
        h = add(h2, sc0);

        let h3 = groupNorm(h, w["down_blocks.1.resnets.1.norm1.weight"], w["down_blocks.1.resnets.1.norm1.bias"], this.numGroups);
        h3 = applySilu(h3);
        h3 = conv2d(h3, w["down_blocks.1.resnets.1.conv1.weight"], w["down_blocks.1.resnets.1.conv1.bias"], 1, 1);
        h3 = addTime(h3, temb, w, "down_blocks.1.resnets.1");
        h3 = groupNorm(h3, w["down_blocks.1.resnets.1.norm2.weight"], w["down_blocks.1.resnets.1.norm2.bias"], this.numGroups);
        h3 = applySilu(h3);
        h3 = conv2d(h3, w["down_blocks.1.resnets.1.conv2.weight"], w["down_blocks.1.resnets.1.conv2.bias"], 1, 1);
        h = add(h3, h);
        const skip2 = cloneFeat(h);

        h = conv2d(h, w["down_blocks.1.downsamplers.0.conv.weight"], w["down_blocks.1.downsamplers.0.conv.bias"], 2, 1);

        // ----- down block 2 (bottom) -----
        let h4 = groupNorm(h, w["down_blocks.2.resnets.0.norm1.weight"], w["down_blocks.2.resnets.0.norm1.bias"], this.numGroups);
        h4 = applySilu(h4);
        h4 = conv2d(h4, w["down_blocks.2.resnets.0.conv1.weight"], w["down_blocks.2.resnets.0.conv1.bias"], 1, 1);
        h4 = addTime(h4, temb, w, "down_blocks.2.resnets.0");
        h4 = groupNorm(h4, w["down_blocks.2.resnets.0.norm2.weight"], w["down_blocks.2.resnets.0.norm2.bias"], this.numGroups);
        h4 = applySilu(h4);
        h4 = conv2d(h4, w["down_blocks.2.resnets.0.conv2.weight"], w["down_blocks.2.resnets.0.conv2.bias"], 1, 1);
        const sc1 = conv2d(h, w["down_blocks.2.resnets.0.conv_shortcut.weight"], w["down_blocks.2.resnets.0.conv_shortcut.bias"]);
        h = add(h4, sc1);

        let h5 = groupNorm(h, w["down_blocks.2.resnets.1.norm1.weight"], w["down_blocks.2.resnets.1.norm1.bias"], this.numGroups);
        h5 = applySilu(h5);
        h5 = conv2d(h5, w["down_blocks.2.resnets.1.conv1.weight"], w["down_blocks.2.resnets.1.conv1.bias"], 1, 1);
        h5 = addTime(h5, temb, w, "down_blocks.2.resnets.1");
        h5 = groupNorm(h5, w["down_blocks.2.resnets.1.norm2.weight"], w["down_blocks.2.resnets.1.norm2.bias"], this.numGroups);
        h5 = applySilu(h5);
        h5 = conv2d(h5, w["down_blocks.2.resnets.1.conv2.weight"], w["down_blocks.2.resnets.1.conv2.bias"], 1, 1);
        h = add(h5, h);

        // ----- mid block (no attention) -----
        let m0 = groupNorm(h, w["mid_block.resnets.0.norm1.weight"], w["mid_block.resnets.0.norm1.bias"], this.numGroups);
        m0 = applySilu(m0);
        m0 = conv2d(m0, w["mid_block.resnets.0.conv1.weight"], w["mid_block.resnets.0.conv1.bias"], 1, 1);
        m0 = addTime(m0, temb, w, "mid_block.resnets.0");
        m0 = groupNorm(m0, w["mid_block.resnets.0.norm2.weight"], w["mid_block.resnets.0.norm2.bias"], this.numGroups);
        m0 = applySilu(m0);
        m0 = conv2d(m0, w["mid_block.resnets.0.conv2.weight"], w["mid_block.resnets.0.conv2.bias"], 1, 1);
        h = add(m0, h);

        let m1 = groupNorm(h, w["mid_block.resnets.1.norm1.weight"], w["mid_block.resnets.1.norm1.bias"], this.numGroups);
        m1 = applySilu(m1);
        m1 = conv2d(m1, w["mid_block.resnets.1.conv1.weight"], w["mid_block.resnets.1.conv1.bias"], 1, 1);
        m1 = addTime(m1, temb, w, "mid_block.resnets.1");
        m1 = groupNorm(m1, w["mid_block.resnets.1.norm2.weight"], w["mid_block.resnets.1.norm2.bias"], this.numGroups);
        m1 = applySilu(m1);
        m1 = conv2d(m1, w["mid_block.resnets.1.conv2.weight"], w["mid_block.resnets.1.conv2.bias"], 1, 1);
        h = add(m1, h);

        // ----- up block 0 -----
        h = convTranspose2d(h, w["up_blocks.0.upsamplers.0.conv.weight"], w["up_blocks.0.upsamplers.0.conv.bias"], 2, 1);
        // concat skip2
        for (let c = 0; c < skip2.length; c++) {
            h.push(skip2[c].map(row => [...row]));
        }

        let u0 = groupNorm(h, w["up_blocks.0.resnets.0.norm1.weight"], w["up_blocks.0.resnets.0.norm1.bias"], this.numGroups);
        u0 = applySilu(u0);
        u0 = conv2d(u0, w["up_blocks.0.resnets.0.conv1.weight"], w["up_blocks.0.resnets.0.conv1.bias"], 1, 1);
        u0 = addTime(u0, temb, w, "up_blocks.0.resnets.0");
        u0 = groupNorm(u0, w["up_blocks.0.resnets.0.norm2.weight"], w["up_blocks.0.resnets.0.norm2.bias"], this.numGroups);
        u0 = applySilu(u0);
        u0 = conv2d(u0, w["up_blocks.0.resnets.0.conv2.weight"], w["up_blocks.0.resnets.0.conv2.bias"], 1, 1);
        const sc2 = conv2d(h, w["up_blocks.0.resnets.0.conv_shortcut.weight"], w["up_blocks.0.resnets.0.conv_shortcut.bias"]);
        h = add(u0, sc2);

        let u1 = groupNorm(h, w["up_blocks.0.resnets.1.norm1.weight"], w["up_blocks.0.resnets.1.norm1.bias"], this.numGroups);
        u1 = applySilu(u1);
        u1 = conv2d(u1, w["up_blocks.0.resnets.1.conv1.weight"], w["up_blocks.0.resnets.1.conv1.bias"], 1, 1);
        u1 = addTime(u1, temb, w, "up_blocks.0.resnets.1");
        u1 = groupNorm(u1, w["up_blocks.0.resnets.1.norm2.weight"], w["up_blocks.0.resnets.1.norm2.bias"], this.numGroups);
        u1 = applySilu(u1);
        u1 = conv2d(u1, w["up_blocks.0.resnets.1.conv2.weight"], w["up_blocks.0.resnets.1.conv2.bias"], 1, 1);
        h = add(u1, h);

        // ----- up block 1 -----
        h = convTranspose2d(h, w["up_blocks.1.upsamplers.0.conv.weight"], w["up_blocks.1.upsamplers.0.conv.bias"], 2, 1);
        // concat skip1
        for (let c = 0; c < skip1.length; c++) {
            h.push(skip1[c].map(row => [...row]));
        }

        let u2 = groupNorm(h, w["up_blocks.1.resnets.0.norm1.weight"], w["up_blocks.1.resnets.0.norm1.bias"], this.numGroups);
        u2 = applySilu(u2);
        u2 = conv2d(u2, w["up_blocks.1.resnets.0.conv1.weight"], w["up_blocks.1.resnets.0.conv1.bias"], 1, 1);
        u2 = addTime(u2, temb, w, "up_blocks.1.resnets.0");
        u2 = groupNorm(u2, w["up_blocks.1.resnets.0.norm2.weight"], w["up_blocks.1.resnets.0.norm2.bias"], this.numGroups);
        u2 = applySilu(u2);
        u2 = conv2d(u2, w["up_blocks.1.resnets.0.conv2.weight"], w["up_blocks.1.resnets.0.conv2.bias"], 1, 1);
        const sc3 = conv2d(h, w["up_blocks.1.resnets.0.conv_shortcut.weight"], w["up_blocks.1.resnets.0.conv_shortcut.bias"]);
        h = add(u2, sc3);

        let u3 = groupNorm(h, w["up_blocks.1.resnets.1.norm1.weight"], w["up_blocks.1.resnets.1.norm1.bias"], this.numGroups);
        u3 = applySilu(u3);
        u3 = conv2d(u3, w["up_blocks.1.resnets.1.conv1.weight"], w["up_blocks.1.resnets.1.conv1.bias"], 1, 1);
        u3 = addTime(u3, temb, w, "up_blocks.1.resnets.1");
        u3 = groupNorm(u3, w["up_blocks.1.resnets.1.norm2.weight"], w["up_blocks.1.resnets.1.norm2.bias"], this.numGroups);
        u3 = applySilu(u3);
        u3 = conv2d(u3, w["up_blocks.1.resnets.1.conv2.weight"], w["up_blocks.1.resnets.1.conv2.bias"], 1, 1);
        h = add(u3, h);

        // ----- up block 2 (head) -----
        let u4 = groupNorm(h, w["up_blocks.2.resnets.0.norm1.weight"], w["up_blocks.2.resnets.0.norm1.bias"], this.numGroups);
        u4 = applySilu(u4);
        u4 = conv2d(u4, w["up_blocks.2.resnets.0.conv1.weight"], w["up_blocks.2.resnets.0.conv1.bias"], 1, 1);
        u4 = addTime(u4, temb, w, "up_blocks.2.resnets.0");
        u4 = groupNorm(u4, w["up_blocks.2.resnets.0.norm2.weight"], w["up_blocks.2.resnets.0.norm2.bias"], this.numGroups);
        u4 = applySilu(u4);
        u4 = conv2d(u4, w["up_blocks.2.resnets.0.conv2.weight"], w["up_blocks.2.resnets.0.conv2.bias"], 1, 1);
        // optional shortcut
        let sc4;
        if (w["up_blocks.2.resnets.0.conv_shortcut.weight"]) {
            sc4 = conv2d(h, w["up_blocks.2.resnets.0.conv_shortcut.weight"], w["up_blocks.2.resnets.0.conv_shortcut.bias"]);
        } else {
            sc4 = u4;
        }
        h = add(u4, sc4);

        let u5 = groupNorm(h, w["up_blocks.2.resnets.1.norm1.weight"], w["up_blocks.2.resnets.1.norm1.bias"], this.numGroups);
        u5 = applySilu(u5);
        u5 = conv2d(u5, w["up_blocks.2.resnets.1.conv1.weight"], w["up_blocks.2.resnets.1.conv1.bias"], 1, 1);
        u5 = addTime(u5, temb, w, "up_blocks.2.resnets.1");
        u5 = groupNorm(u5, w["up_blocks.2.resnets.1.norm2.weight"], w["up_blocks.2.resnets.1.norm2.bias"], this.numGroups);
        u5 = applySilu(u5);
        u5 = conv2d(u5, w["up_blocks.2.resnets.1.conv2.weight"], w["up_blocks.2.resnets.1.conv2.bias"], 1, 1);
        h = add(u5, h);

        // ----- out head -----
        let hout = groupNorm(h, w["conv_norm_out.weight"], w["conv_norm_out.bias"], this.numGroups);
        hout = applySilu(hout);
        const out = conv2d(hout, w["conv_out.weight"], w["conv_out.bias"], 1, 1);
        return out[0];  // 1x28x28
    }
}

function applySilu(feat) {
    const C = feat.length;
    const H = feat[0].length;
    const W = feat[0][0].length;
    
    for (let c = 0; c < C; c++) {
        for (let i = 0; i < H; i++) {
            for (let j = 0; j < W; j++) {
                feat[c][i][j] = silu(feat[c][i][j]);
            }
        }
    }
    return feat;
}

function add(a, b) {
    const C = a.length;
    const H = a[0].length;
    const W = a[0][0].length;
    
    for (let c = 0; c < C; c++) {
        for (let i = 0; i < H; i++) {
            for (let j = 0; j < W; j++) {
                a[c][i][j] += b[c][i][j];
            }
        }
    }
    return a;
}

function cloneFeat(f) {
    return f.map(ch => ch.map(row => [...row]));
}

function addTime(h, temb, w, prefix) {
    const tW = w[`${prefix}.time_emb_proj.weight`];
    const tB = w[`${prefix}.time_emb_proj.bias`];
    const outCh = tW.length;
    
    const proj = [];
    for (let c = 0; c < outCh; c++) {
        let sum = tB[c];
        for (let k = 0; k < tW[c].length; k++) {
            sum += temb[k] * tW[c][k];
        }
        proj[c] = sum;
    }
    
    const H = h[0].length;
    const W = h[0][0].length;
    
    for (let c = 0; c < outCh; c++) {
        const val = proj[c];
        for (let i = 0; i < H; i++) {
            for (let j = 0; j < W; j++) {
                h[c][i][j] += val;
            }
        }
    }
    return h;
}

// Image loading and saving functions
function loadReferenceImage() {
    try {
        if (fs.existsSync('ref_prepare_result.png')) {
            const data = fs.readFileSync('ref_prepare_result.png');
            const png = PNG.sync.read(data);
            
            // Convert PNG data to 2D array, normalized to [-1, 1] range
            const img = [];
            for (let y = 0; y < 28; y++) {
                const row = [];
                for (let x = 0; x < 28; x++) {
                    const idx = (28 * y + x) << 2; // multiply by 4 for RGBA
                    // Use grayscale value (R channel), convert from [0,255] to [-1,1]
                    const grayValue = png.data[idx] / 127.5 - 1.0;
                    row.push(grayValue);
                }
                img.push(row);
            }
            console.log("Loaded reference image from ref_prepare_result.png");
            return img;
        } else {
            throw new Error("Reference image not found");
        }
    } catch (error) {
        console.log("Warning: Could not load reference image, using random noise");
        const img = [];
        for (let i = 0; i < 28; i++) {
            const row = [];
            for (let j = 0; j < 28; j++) {
                // Generate random noise with Gaussian distribution
                row.push(gaussianRandom(0.0, 1.0));
            }
            img.push(row);
        }
        return img;
    }
}

// Box-Muller transform for Gaussian random numbers
function gaussianRandom(mean = 0, std = 1) {
    if (gaussianRandom.hasSpare) {
        gaussianRandom.hasSpare = false;
        return gaussianRandom.spare * std + mean;
    }
    
    gaussianRandom.hasSpare = true;
    
    const u = Math.random();
    const v = Math.random();
    const mag = std * Math.sqrt(-2.0 * Math.log(u));
    gaussianRandom.spare = mag * Math.cos(2.0 * Math.PI * v);
    
    return mag * Math.sin(2.0 * Math.PI * v) + mean;
}

function saveImageToPNG(imageData, filename) {
    try {
        if (!imageData || !imageData.length || !imageData[0] || !imageData[0].length) {
            throw new Error("Invalid image data");
        }
        
        const height = imageData.length;
        const width = imageData[0].length;
        
        // Create PNG with actual dimensions
        const png = new PNG({ width: width, height: height, colorType: 0 }); // colorType 0 = grayscale
        
        // Find min/max for normalization
        let mn = Infinity, mx = -Infinity;
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                mn = Math.min(mn, imageData[i][j]);
                mx = Math.max(mx, imageData[i][j]);
            }
        }
        
        // Convert 2D array to PNG data
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (width * y + x) << 2; // multiply by 4 for RGBA
                
                let pixelValue;
                if (mx - mn > 1e-8) {
                    // Normalize from [mn,mx] to [0,255]
                    pixelValue = Math.floor(((imageData[y][x] - mn) / (mx - mn)) * 255);
                } else {
                    pixelValue = 128; // Gray if all values are the same
                }
                
                pixelValue = Math.max(0, Math.min(255, pixelValue));
                
                // Set RGBA values (grayscale, so R=G=B)
                png.data[idx] = pixelValue;     // R
                png.data[idx + 1] = pixelValue; // G
                png.data[idx + 2] = pixelValue; // B
                png.data[idx + 3] = 255;        // A (fully opaque)
            }
        }
        
        // Write PNG to file
        const buffer = PNG.sync.write(png);
        fs.writeFileSync(`${filename}.png`, buffer);
        console.log(`Saved ${filename}.png`);
    } catch (error) {
        console.error(`Error saving PNG: ${error.message}`);
        // Fallback to text representation
        const textData = imageData.map(row => 
            row.map(pixel => Math.round(pixel * 255).toString().padStart(3)).join(' ')
        ).join('\n');
        
        fs.writeFileSync(`${filename}.txt`, textData);
        console.log(`Saved ${filename}.txt (text representation)`);
    }
}

// ---------- Main (DDPM Sampling) ----------

async function main() {
    try {
        // Load the data
        const weightsData = fs.readFileSync('unet_weights.json', 'utf8');
        const weights = JSON.parse(weightsData);
        
        const betasData = fs.readFileSync('betas.json', 'utf8');
        const betas = JSON.parse(betasData);

        // Precompute alphas / alpha_cumprod
        const alphas = betas.map(b => 1.0 - b);
        const alphaBar = [];
        let prod = 1.0;
        for (let t = 0; t < alphas.length; t++) {
            prod *= alphas[t];
            alphaBar[t] = prod;
        }

        const model = new SimpleUNet(weights);

        // Use reference image as starting point instead of pure random noise
        console.log("Loading reference image for guidance...");
        const refImg = loadReferenceImage();
        
        // Add some controlled noise to the reference image to allow for variation
        const noiseStrength = 0.5;  // Adjust this to control how much we deviate from reference
        let img = [];
        for (let i = 0; i < 28; i++) {
            const row = [];
            for (let j = 0; j < 28; j++) {
                row.push(refImg[i][j] + noiseStrength * gaussianRandom(0.0, 1.0));
            }
            img.push(row);
        }

        // Use more steps for better quality
        const numSteps = 1;
        
        // Create a better timestep schedule - use the actual scheduler timesteps
        const totalTimesteps = betas.length;
        let tGrid;
        if (numSteps > 1) {
            // Use evenly spaced timesteps from the full range
            const stepSize = Math.floor(totalTimesteps / numSteps);
            tGrid = [];
            for (let i = 0; i < numSteps; i++) {
                tGrid.push(totalTimesteps - 1 - i * stepSize);
            }
            tGrid[tGrid.length - 1] = 0;  // Ensure we end at t=0
        } else {
            tGrid = [totalTimesteps - 1];
        }

        console.log(`Running DDPM sampling with ${numSteps} steps...`);
        const frames = [];

        for (let stepIdx = 0; stepIdx < tGrid.length; stepIdx++) {
            let t = tGrid[stepIdx];
            // Ensure t is within bounds
            t = Math.max(0, Math.min(t, betas.length - 1));
            
            // --- time embedding ---
            const embInDim = weights["time_embedding.linear_1.weight"][0].length;
            const half = Math.floor(embInDim / 2);
            
            // sinusoidal embedding
            const sinu = [];
            for (let i = 0; i < half; i++) {
                const freq = Math.exp(-Math.log(10000) * i / Math.max(half - 1, 1));
                sinu.push(Math.cos(t * freq));
            }
            for (let i = 0; i < half; i++) {
                const freq = Math.exp(-Math.log(10000) * i / Math.max(half - 1, 1));
                sinu.push(Math.sin(t * freq));
            }
            
            // linear_1
            const l1w = weights["time_embedding.linear_1.weight"];
            const l1b = weights["time_embedding.linear_1.bias"];
            const h = [];
            for (let o = 0; o < l1w.length; o++) {
                let sum = l1b[o];
                for (let i = 0; i < sinu.length; i++) {
                    sum += sinu[i] * l1w[o][i];
                }
                h.push(silu(sum));
            }
            
            // linear_2
            const l2w = weights["time_embedding.linear_2.weight"];
            const l2b = weights["time_embedding.linear_2.bias"];
            const temb = [];
            for (let o = 0; o < l2w.length; o++) {
                let sum = l2b[o];
                for (let i = 0; i < h.length; i++) {
                    sum += h[i] * l2w[o][i];
                }
                temb.push(silu(sum));
            }

            // --- Predict noise eps_theta(x_t, t) ---
            const xIn = [img];  // (C=1,H,W)
            const eps = model.predictNoise(xIn, t, temb);
            
            // Verify eps has the right dimensions (should be 28x28 now)
            if (eps.length !== 28 || eps[0].length !== 28) {
                console.log(`Error: eps shape is ${eps.length}x${eps[0].length}, expected 28x28`);
                console.log("Check the convTranspose2d output_padding logic");
                // Continue with whatever dimensions we have for testing
            }

            // --- DDPM reverse update with better numerical stability ---
            if (t > 0) {
                const aT = alphas[t];
                const abT = alphaBar[t];
                
                // Use previous timestep's alpha_bar for better computation
                const tPrev = stepIdx + 1 < tGrid.length ? tGrid[stepIdx + 1] : 0;
                const abPrev = tPrev > 0 ? alphaBar[tPrev] : 1.0;
                
                // Compute coefficients with better numerical stability
                const sqrtInvAt = aT > 1e-8 ? 1.0 / Math.sqrt(aT) : 1.0;
                const sqrtAbT = Math.sqrt(Math.max(1e-12, abT));
                const sqrtOneMinusAb = Math.sqrt(Math.max(1e-12, 1.0 - abT));
                
                // Mean computation
                const coeff = (1.0 - aT) / sqrtOneMinusAb;
                const mean = [];
                const H = Math.min(img.length, eps.length);
                const W = Math.min(img[0].length, eps[0].length);
                for (let i = 0; i < H; i++) {
                    const row = [];
                    for (let j = 0; j < W; j++) {
                        row.push(sqrtInvAt * (img[i][j] - coeff * eps[i][j]));
                    }
                    mean.push(row);
                }
                
                // Variance computation - use the correct DDPM formula
                const betaT = betas[t];
                const posteriorVar = (1.0 - abT) > 1e-12 ? betaT * (1.0 - abPrev) / (1.0 - abT) : 0.0;
                const sigma = Math.sqrt(Math.max(0.0, posteriorVar));
                
                // Add noise if not the last step
                if (stepIdx < tGrid.length - 1 && sigma > 0) {
                    const noise = [];
                    for (let i = 0; i < H; i++) {
                        const row = [];
                        for (let j = 0; j < W; j++) {
                            row.push(gaussianRandom(0.0, 1.0));
                        }
                        noise.push(row);
                    }
                    
                    // Update img to match the dimensions of mean
                    img = [];
                    for (let i = 0; i < H; i++) {
                        const row = [];
                        for (let j = 0; j < W; j++) {
                            row.push(mean[i][j] + sigma * noise[i][j]);
                        }
                        img.push(row);
                    }
                } else {
                    // Update img to match the dimensions of mean
                    img = [];
                    for (let i = 0; i < H; i++) {
                        const row = [];
                        for (let j = 0; j < W; j++) {
                            row.push(mean[i][j]);
                        }
                        img.push(row);
                    }
                }
            } else {
                // Final step - deterministic
                const aT = alphas.length > 0 ? alphas[0] : 1.0;
                const sqrtInvAt = aT > 1e-8 ? 1.0 / Math.sqrt(aT) : 1.0;
                const sqrtOneMinusAb = Math.sqrt(Math.max(1e-12, 1.0 - alphaBar[0]));
                const coeff = sqrtOneMinusAb;
                
                const H = Math.min(img.length, eps.length);
                const W = Math.min(img[0].length, eps[0].length);
                const finalImg = [];
                for (let i = 0; i < H; i++) {
                    const row = [];
                    for (let j = 0; j < W; j++) {
                        row.push(sqrtInvAt * (img[i][j] - coeff * eps[i][j]));
                    }
                    finalImg.push(row);
                }
                img = finalImg;
            }

            // Record frame
            frames.push(img.map(row => [...row]));
            
            if (stepIdx % 10 === 0) {
                console.log(`Step ${stepIdx + 1}/${numSteps} (t=${t}) complete`);
            }
        }

        console.log("Sampling complete, creating output images...");

        // Save final result
        const finalFrame = frames[frames.length - 1];
        saveImageToPNG(finalFrame, "result_js_final");
        
    } catch (error) {
        console.error("Error:", error);
    }
}

if (require.main === module) {
    main();
}