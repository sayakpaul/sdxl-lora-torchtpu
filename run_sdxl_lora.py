# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "torch==2.14.0", "torchvision==0.29.0", "torch-tpu==0.3.0",
#   "diffusers==0.42.1", "accelerate==1.15.0", "transformers==5.17.0",
#   "peft==0.19.0", "trackio==0.45.1", "ftfy", "jinja2",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# ///
"""Run the Diffusers SDXL DreamBooth LoRA example on one native TorchTPU device."""

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4


MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SCRIPT = "train_dreambooth_lora_sdxl.py"
REVISION = "2a1433913c7044f34fa4767dd84b28bba0bb0ecf"  # Diffusers v0.42.1
SOURCE_URL = f"https://raw.githubusercontent.com/huggingface/diffusers/{REVISION}/examples/dreambooth/{SCRIPT}"
SOURCE_SHA256 = "9bb7fe75fa1646ddfe41a028796baff6f7ff3f4b48f57b664526c4e3f71b37bb"
ADAPTER = "pytorch_lora_weights.safetensors"
TRACKIO_PROJECT = "dreambooth-lora-sd-xl"
RUN_INFO = "trackio-run.json"


def prepare_trainer(directory, reference):
    if reference:
        source = reference.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=60) as response:
            source = response.read()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError(f"Expected the Diffusers v0.42.1 example from {SOURCE_URL}")
    trainer = directory / SCRIPT
    trainer.write_bytes(source)
    with Path(__file__).with_name("sdxl-torchtpu.patch").open("rb") as patch:
        subprocess.run(["patch", "-s", "-p1"], cwd=directory, stdin=patch, check=True)
    return trainer


def prepare_images(destination, source):
    if source is None:
        from huggingface_hub import snapshot_download

        source = Path(
            snapshot_download("diffusers/dog-example", repo_type="dataset", allow_patterns=["*.jpeg", "*.jpg"])
        )
    extensions = {".jpeg", ".jpg", ".png", ".webp", ".bmp"}
    images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not images:
        raise ValueError(f"No instance images in {source}")
    # The reference dataset opens every directory entry, including Hub metadata.
    destination.mkdir()
    for image in images:
        (destination / image.name).symlink_to(image.resolve())
    return destination


def verify_adapter(output):
    import torch
    from safetensors.torch import load_file

    path = output / ADAPTER
    state = load_file(path)
    if not state or not all(key.startswith("unet.") and torch.isfinite(value).all() for key, value in state.items()):
        raise ValueError(f"Expected finite U-Net adapter weights in {path}")
    up = [v for k, v in state.items() if k.endswith((".up.weight", ".lora_B.weight"))]
    if not any(torch.count_nonzero(v) for v in up):
        raise ValueError("The zero-initialized LoRA output projections did not update")
    print(f"Verified {len(state)} adapter tensors in {path}", flush=True)


def preview(args, device):
    import torch
    import trackio
    from diffusers import StableDiffusionXLPipeline

    verify_adapter(args.output)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, variant="fp16", torch_dtype=torch.bfloat16)
    pipe.to(device)
    prompt = args.prompt or "a photo of sks dog wearing a red scarf"
    latents = {}
    for name in ("before", "after"):
        if name == "after":
            pipe.load_lora_weights(str(args.output))
        latents[name] = pipe(
            prompt,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=25,
            generator=torch.Generator(device="cpu").manual_seed(0),
            output_type="latent",
        ).images.to(device="cpu", dtype=torch.float32)

    # TorchTPU's decoder GroupNorm cannot view the channels-last activation.
    # Denoise both images on TPU, then decode the two final latents on the host.
    vae = pipe.vae.to(device="cpu", dtype=torch.float32)
    media = {}
    for name, latent in latents.items():
        with torch.inference_mode():
            pixels = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
        if pipe.watermark is not None:
            pixels = pipe.watermark.apply_watermark(pixels)
        image = pipe.image_processor.postprocess(pixels, output_type="pil")[0]
        path = args.output / f"{name}.png"
        image.save(path)
        media[f"preview/{name}"] = trackio.Image(image, caption=f"{name} | {prompt} | seed 0")
        print(f"Saved {path}", flush=True)
    info = args.output / RUN_INFO
    if info.exists():
        run = json.loads(info.read_text())
        history = next(r for r in trackio.Api().runs(run["project"]) if r.name == run["name"])
        # Trackio writes the supplied config again when the resumed run logs media.
        config = {k: v for k, v in (history.config or {}).items() if not k.startswith("_")}
        trackio.init(**run, config=config, resume="must", auto_log_gpu=False)
    else:
        # Adapters from earlier versions have no associated training history.
        trackio.init(project=TRACKIO_PROJECT, name=f"{args.output.name}-preview", auto_log_gpu=False)
    try:
        trackio.log(media)
    finally:
        trackio.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, help="flat folder of instance photos (default: diffusers/dog-example)")
    parser.add_argument("--prompt", help="instance prompt for training, or generation prompt with --preview-only")
    parser.add_argument("--steps", type=int, default=100, help="optimizer updates (default: 100)")
    parser.add_argument("--resolution", type=int, choices=(512, 768, 1024), default=512)
    parser.add_argument("--output", type=Path, default=Path("sdxl-dog-lora"))
    parser.add_argument("--compile-blocks", action="store_true", help="compile repeated U-Net transformer blocks")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="trade recomputation for activation memory")
    parser.add_argument("--reference-script", type=Path, help="use a local copy of the pinned upstream trainer")
    parser.add_argument("--preview-only", action="store_true", help="render before.png and after.png from a saved run")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    os.environ["TRACKIO_DIR"] = str((args.output / "trackio").resolve())
    os.environ["ACCELERATE_TORCH_DEVICE"] = "tpu:0"
    import torch
    import torch_tpu  # Registers the native device when PyTorch autoload is disabled.
    from accelerate import PartialState

    state = PartialState()
    if state.device.type != "tpu" or state.num_processes != 1:
        raise RuntimeError(f"Expected a single TorchTPU process, got {state}")
    probe = torch.ones(1, device=state.device, dtype=torch.bfloat16)
    print(f"Device {probe.device}, BF16 probe {probe.item()}, PyTorch {torch.__version__}", flush=True)
    if args.preview_only:
        preview(args, state.device)
        return

    os.environ["SDXL_COMPILE_BLOCKS"] = str(int(args.compile_blocks))
    run_name = f"{args.output.name}-{uuid4().hex[:12]}"
    os.environ["SDXL_TRACKIO_RUN_NAME"] = run_name
    with tempfile.TemporaryDirectory(prefix="sdxl-dreambooth-") as temp:
        work = Path(temp)
        trainer = prepare_trainer(work, args.reference_script)
        images = prepare_images(work / "images", args.images)
        sys.argv = [
            str(trainer),
            f"--pretrained_model_name_or_path={MODEL}",
            "--variant=fp16",
            f"--instance_data_dir={images}",
            f"--instance_prompt={args.prompt or 'a photo of sks dog'}",
            f"--output_dir={args.output}",
            f"--resolution={args.resolution}",
            "--center_crop",
            "--rank=4",
            "--report_to=trackio",
            "--train_batch_size=1",
            "--gradient_accumulation_steps=1",
            "--learning_rate=1e-4",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            f"--max_train_steps={args.steps}",
            f"--checkpointing_steps={args.steps + 1}",
            "--mixed_precision=no",
            "--seed=0",
        ]
        if args.gradient_checkpointing or args.resolution > 512:
            sys.argv.append("--gradient_checkpointing")
        runpy.run_path(str(trainer), run_name="__main__")
    verify_adapter(args.output)
    (args.output / RUN_INFO).write_text(json.dumps({"project": TRACKIO_PROJECT, "name": run_name}) + "\n")


if __name__ == "__main__":
    main()
