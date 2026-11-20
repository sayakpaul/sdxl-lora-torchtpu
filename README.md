# LoRA fine-tuning SDXL on TPUs using Diffusers and TorchTPU

Shows how to fine-tune SDXL using LoRA and the changes needed to make it run on TPUs when using TorchTPU and Diffusers.

## What changed for TPU

We start by making some changes to the [existing
training script](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md) in Diffusers:

- Accelerate runs the U-Net on `tpu:0` through TorchTPU. Frozen weights use BF16, while the LoRA weights and AdamW state use FP32.
- Before training, the CPU encodes the prompt and caches each photo’s VAE mean and standard deviation. The TPU samples a fresh latent at each update.
- The Colab v5e1 run uses gradient checkpointing to fit in memory. For previews, the TPU denoises and the CPU decodes the final images.

Before training, the launcher applies the [TorchTPU patch](sdxl-torchtpu.patch) to the Diffusers v0.42.1 SDXL DreamBooth script to cache encoder outputs on the CPU and train the LoRA weights on the TPU.

## Run

From the repo root, use an authenticated `gcloud` CLI with TPU quota in your active project, or `uvx` with [Colab-authorized application default credentials](https://github.com/googlecolab/google-colab-cli#global-options).

```bash
# Google Cloud · one v6e-1
ZONE=us-east5-b ./run_on_tpu.sh

# Colab · one v5e1
COLAB_TPU=v5e1 ./run_on_colab.sh
```

Both launchers save the adapter, previews, and Trackio logs under `runs/<job>/result/`.

## Hyperparameters

Both launchers upload the same trainer. Edit these settings before launching. [run_sdxl_lora.py](run_sdxl_lora.py)

| Where to edit | Settings and defaults |
| --- | --- |
| Argument parser | `--steps` = 100, `--resolution` = 512 (also accepts 768 or 1024) |
| Diffusers argument list | `--rank` = 4, `--learning_rate` = 1e-4, `--train_batch_size` = 1 |

## Timing

At 512 × 512 and batch size 1, 100 training updates took **62 seconds on GCP v6e-1** and **143 seconds on Colab v5e1**. These times exclude setup and preview generation.

## Before and after

Both images use the prompt “a photo of sks dog wearing a red scarf,” seed 0, and 25 denoising steps on Colab v5e1.

| Before · SDXL Base | After · 100 LoRA updates |
| --- | --- |
| ![Base SDXL output for the red-scarf prompt][before] | ![Fine-tuned SDXL output for the same prompt and seed][after] |

[before]: https://huggingface.co/buckets/sayakpaul/torchtpu-sdxl-lora-sprint-bucket/resolve/media/Colab-v5e1/101/0160bcb2-c187-4d72-a9b8-d9701a3a9fbb.png
[after]: https://huggingface.co/buckets/sayakpaul/torchtpu-sdxl-lora-sprint-bucket/resolve/media/Colab-v5e1/101/f81d8462-c142-4884-b21b-3b1650c39ee5.png

[View both loss curves and all four previews on Trackio](https://huggingface.co/spaces/sayakpaul/torchtpu-sdxl-lora-sprint).

## Inference on CUDA

Load either TPU-trained adapter into SDXL Base on a CUDA GPU. Both were verified on an NVIDIA A10G. [Compare the GPU outputs](https://huggingface.co/sayakpaul/torchtpu-sdxl-lora-sprint#gpu-inference-verification)

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", torch_dtype=torch.float16
).to("cuda")
pipe.load_lora_weights(
    "sayakpaul/torchtpu-sdxl-lora-sprint", revision="sprint-v1",
    subfolder="colab-v5e1",  # or "gcp-v6e-1"
    weight_name="pytorch_lora_weights.safetensors",
)

image = pipe(
    "a photo of sks dog wearing a red scarf", height=512, width=512,
    num_inference_steps=25, guidance_scale=5.0, generator=torch.Generator().manual_seed(0),
).images[0]
image.save("dog.png")
```
