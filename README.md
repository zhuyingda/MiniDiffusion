# MiniDDPM
a mini DDPM implementation based on ddpm-mnist[https://huggingface.co/1aurent/ddpm-mnist] with a `28x28` image generation.

## usage

You can generate an image by running these commands:

```
python -m venv .venv
pip install -r requirements.txt
python prepare.py
python extract_weights.py
python unet.py
```

You will get `result_final.png` as the output result by `unet.py`.

## based datasets

- [1aurent/ddpm-mnist](https://huggingface.co/1aurent/ddpm-mnist)
- [ylecun/minist](https://huggingface.co/datasets/ylecun/mnist)