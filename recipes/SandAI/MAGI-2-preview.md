# MAGI-2 Preview

> Native text/image-to-video-and-audio generation with the released Preview stage

## Summary

- Vendor: SandAI
- Model: `sand-ai/MAGI-2-preview`
- Task: Text-to-video-and-audio (T2VA) and image-to-video-and-audio (I2VA)
- Mode: Offline shared examples and OpenAI-compatible video serving
- Runtime: Native vLLM-Omni pipeline
- Maintainer: Community

## When to use this recipe

Use this recipe to run MAGI-2 Preview directly through vLLM-Omni. The native
pipeline owns model construction, checkpoint loading, sampling, Ulysses
sequence parallelism, image conditioning, and video/audio decoding.

The supported native scope is the released Preview stage at `272p` or `540p`.
Both modes produce a 10-second clip with 125 frames at 12.5 fps and synchronized
44.1 kHz stereo audio.

The commands below reuse the shared
[`text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py)
and
[`image_to_video.py`](../../examples/offline_inference/image_to_video/image_to_video.py)
entrypoints. Do not create a model-specific example script.

## References

- Model card: <https://huggingface.co/sand-ai/MAGI-2-preview>
- Supported-model table: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Diffusion feature table: [`docs/user_guide/diffusion_features.md`](../../docs/user_guide/diffusion_features.md)

## Prepare the Preview checkpoint

The native pipeline requires the Preview transformer, text encoder, video and
audio decoders, and VAE assets. These directories occupy approximately 274 GiB
on disk. Download them to a local directory for predictable multi-worker
startup:

```bash
export MAGI2_CKPT_ROOT=/path/to/MAGI-2-preview

hf download sand-ai/MAGI-2-preview \
  preview/ text_encoder/ vae/ turbo_vae/ stable-audio-open-1.0/ \
  --revision 2dea51b64db47ee5b4402d36fd90829a0c58913b \
  --local-dir "$MAGI2_CKPT_ROOT"
```

Install vLLM-Omni before running the examples. The native path uses
vLLM-Omni's normal dependencies and does not require a second model runtime.

## Hardware Support

## GPU

### Native 4-GPU profile

MAGI-2 Preview uses tensor parallel size 1. Its Ulysses sequence-parallel group
must span the complete worker world, so a four-GPU run uses
`--tensor-parallel-size 1 --ulysses-degree 4`.

The default recipe does not enable DLO. Each worker keeps its rank-local
Preview shard device-resident for denoising, while the output worker stages the
text encoder and codec components from pinned CPU memory only for the phase
that needs them. The output worker temporarily makes room for prompt encoding
when required by the device capacity.

#### Environment

- Platform: NVIDIA CUDA
- Workers: 4
- Tensor parallel size: 1
- Ulysses degree / sequence parallel size: 4
- Dtype: BF16

Set `MAGI2_DETERMINISTIC=1` before worker startup when deterministic kernels are
required. Deterministic mode is fixed for the lifetime of the workers.

#### Text-to-video-and-audio

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python examples/offline_inference/text_to_video/text_to_video.py \
  --model "$MAGI2_CKPT_ROOT" \
  --model-class-name Magi2Pipeline \
  --prompt "A red fox walks through fresh snow while wind moves the pine branches." \
  --height 512 \
  --width 896 \
  --num-frames 125 \
  --num-inference-steps 100 \
  --fps 12.5 \
  --tensor-parallel-size 1 \
  --ulysses-degree 4 \
  --extra-body '{"seconds":10,"resolution":"540p"}' \
  --output magi2_540p_t2va.mp4
```

#### Image-to-video-and-audio

The shared I2V entrypoint passes the original image to MAGI-2's
aspect-preserving conditioning path instead of stretching it to the output
size.

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model "$MAGI2_CKPT_ROOT" \
  --model-class-name Magi2Pipeline \
  --image /path/to/first_frame.png \
  --prompt "The fox looks up, then walks forward as snow falls around it." \
  --height 512 \
  --width 896 \
  --num-frames 125 \
  --num-inference-steps 100 \
  --fps 12.5 \
  --tensor-parallel-size 1 \
  --ulysses-degree 4 \
  --extra-body '{"seconds":10,"resolution":"540p"}' \
  --output magi2_540p_i2va.mp4
```

For a load-and-kernel smoke test, change `--num-inference-steps 100` to `1`.
The one-step form is not a quality evaluation.

#### 272p variant

Use the same command with these overrides:

```text
--height 256 --width 448 --extra-body '{"seconds":10,"resolution":"272p"}'
```

#### Verification

```bash
ffprobe -v error \
  -show_entries stream=codec_type,width,height,avg_frame_rate,nb_frames,sample_rate,channels \
  -of json magi2_540p_t2va.mp4
```

The output should contain 125 video frames at 896x512 and 12.5 fps, plus stereo
44.1 kHz audio.

### Native 8-GPU profile

The native eight-GPU topology keeps tensor parallel size at 1 and expands
Ulysses to the full eight-worker world. Select all eight devices:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

Then add `--tensor-parallel-size 1 --ulysses-degree 8` to either shared offline
command. For online serving, set `--num-gpus 8 --tensor-parallel-size 1
--ulysses-degree 8`.

Do not set tensor parallel size to the GPU count. MAGI-2 overlaps its native
MoE-head partitioning with world-spanning Ulysses sequence parallelism.

### Distributed layerwise offload status

MAGI-2 Preview supports distributed layerwise offload only in rank-local,
no-AllGather mode. Its MoE heads are already partitioned across the same worker
group as Ulysses, so the workers do not hold interchangeable data-parallel
shards for DLO to reconstruct with AllGather.

Add these flags to either shared offline command:

```text
--enable-distributed-layerwise-offload \
--dlo-no-use-allgather \
--dlo-resident-layers 0
```

The path was exercised end to end on four NVIDIA L20X GPUs at 272p with all 40
transformer blocks streamed. Its deterministic decoded video and audio matched
the non-DLO run exactly. AllGather DLO is rejected during pipeline validation.

## Online serving

This four-worker example uses the same native topology as the offline commands:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve "$MAGI2_CKPT_ROOT" --omni \
  --model-class-name Magi2Pipeline \
  --num-gpus 4 \
  --tensor-parallel-size 1 \
  --ulysses-degree 4 \
  --port 8091
```

```bash
curl -X POST http://localhost:8091/v1/videos/sync \
  -F 'prompt=A red fox walks through fresh snow.' \
  -F 'seconds=10' \
  -F 'size=896x512' \
  -F 'num_frames=125' \
  -F 'fps=12.5' \
  -F 'num_inference_steps=100' \
  -F 'seed=42' \
  -F 'extra_params={"resolution":"540p"}' \
  -o magi2_online.mp4
```

For I2VA, add `-F 'input_reference=@first_frame.png;type=image/png'`.

## MAGI-2 request fields

Common geometry and sampling values use the shared CLI flags. The native
Preview pipeline accepts these model-specific values through `--extra-body`:

| Field | Meaning |
|---|---|
| `seconds` | Output duration; the Preview release supports `10` only. |
| `resolution` | Native generation tier: `272p` or `540p`; default `540p`. |
| `output_width`, `output_height` | Optional final decoded-frame resize; both must be supplied. |
| `deterministic` | Must match `MAGI2_DETERMINISTIC` fixed at worker startup. |

Use `--image` in the shared I2V entrypoint rather than the lower-level
`image_path` extra field.

## Known limitations

- Exactly one request/output is supported at a time.
- Only the published 10-second, 125-frame, 12.5-fps Preview workflow is supported.
- Native generation is limited to the `272p` and `540p` tiers.
- The worker world must contain 4 or 8 GPUs, with tensor parallel size 1 and
  Ulysses degree equal to world size.
- HSDP, cache acceleration, quantization, generic module CPU offload, CFG
  parallelism, ring sequence parallelism, pipeline parallelism, and VAE patch
  parallelism are not supported for this pipeline.
- Distributed layerwise offload requires `--dlo-no-use-allgather`; the
  AllGather mode is incompatible with MAGI-2's rank-local MoE-head shards.
