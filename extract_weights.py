import json
import torch
from diffusers import DDPMPipeline

# Load the pipeline to extract weights
MODEL_ID = "1aurent/ddpm-mnist"
pipeline = DDPMPipeline.from_pretrained(MODEL_ID)
unet = pipeline.unet
scheduler = pipeline.scheduler

def to_pylist(x: torch.Tensor):
    return x.detach().cpu().tolist()

# Extract all UNet weights
weights = {}
for name, param in unet.named_parameters():
    weights[name] = to_pylist(param)

# Extract scheduler betas
betas = scheduler.betas.detach().cpu().numpy().tolist()

# Save weights and betas separately for unet.py compatibility
with open("unet_weights.json", "w") as f:
    json.dump(weights, f)

with open("betas.json", "w") as f:
    json.dump(betas, f)

print("Extracted weights to unet_weights.json and betas.json")