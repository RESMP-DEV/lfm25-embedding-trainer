# Optional NVIDIA Docker workflow

Docker is an optional reproducibility layer. Native `uv` environments remain the reference
workflow, especially for Apple MPS and AMD ROCm. The image uses the project's frozen `uv.lock`
and installs PyTorch's Linux CUDA dependencies; it does not contain an NVIDIA kernel driver.

## Host requirements

- Linux with an NVIDIA GPU and NVIDIA driver 580 or newer (`nvidia-smi` must succeed)
- Docker Engine with Compose
- NVIDIA Container Toolkit configured for Docker

Verify GPU pass-through before building the trainer:

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

The host driver is injected at runtime. Do not install or replace a host NVIDIA driver from
inside this image. A host CUDA Toolkit and `nvcc` are not required: PyTorch supplies the pinned
CUDA user-space libraries, while the image includes the small C toolchain Triton needs for runtime
JIT driver shims.

## Build and probe

```bash
export TRAINER_UID="$(id -u)"
export TRAINER_GID="$(id -g)"
docker compose build trainer

docker compose run --rm --entrypoint python trainer -c \
  'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name())'
```

The build arguments make the container's `trainer` UID and GID match the Linux user that owns
the bind-mounted checkout. Keep those exports in the shell used for both build and run commands;
the defaults are 1000 only for hosts where the owning user already has UID/GID 1000.

The source tree is mounted at `/workspace`. The named `huggingface-cache` volume preserves model
downloads between runs. Training outputs therefore appear directly in the host repository.

## Train

```bash
docker compose run --rm trainer train examples/pairs.jsonl \
  --config configs/cuda.toml \
  --output models/lfm25-my-data \
  --device cuda
```

For a one-step end-to-end check, use `configs/cuda-smoke.toml` instead. To train on data outside
the repository, add a read-only bind mount and refer to its container path:

```bash
docker compose run --rm \
  -v /absolute/path/to/my-data:/datasets:ro \
  trainer train /datasets/pairs.jsonl \
  --config configs/cuda.toml \
  --output models/lfm25-my-data \
  --device cuda
```

For online W&B tracking, export the secret in the shell that invokes Compose. The key is passed
at runtime and is never copied into the image:

```bash
export WANDB_API_KEY=your-key
docker compose run --rm trainer train examples/pairs.jsonl \
  --config configs/cuda-smoke.toml \
  --output models/cuda-smoke \
  --device cuda \
  --wandb-mode online
```

## Why there is no generic ROCm image

ROCm compatibility depends on the exact GPU family, host kernel/driver, and PyTorch build. A
single nominally portable image would overstate that contract. Use the physically validated
native path in [Fine-tuning on AMD ROCm](AMD_ROCM.md); a ROCm container should be published only
after it is tested on each claimed target.
