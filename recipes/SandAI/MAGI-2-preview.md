# MAGI-2 Preview

> Native text/image-to-video-and-audio generation with the released Preview stage

## Summary

- Vendor: SandAI
- Model: `sand-ai/MAGI-2-preview`
- Task: Text-to-video-and-audio (T2VA) and image-to-video-and-audio (I2VA)
- Mode: Offline shared examples and OpenAI-compatible video serving
- Runtime: Native vLLM-Omni pipeline
- Default deployment: Four-GPU resident SP4 (`TP=1`, `SP=4`)
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

## Hardware support

### GPU

#### Default four-GPU deployment: resident SP4

The recommended default is resident sequence parallelism across all four
workers: `--tensor-parallel-size 1 --ulysses-degree 4`. This SP4 layout is the
reference-aligned fidelity baseline. DLO is disabled, and every worker keeps
its rank-local Preview transformer shard device-resident during denoising.

The output worker stages the text encoder and codec components from pinned CPU
memory only for the phase that needs them. It temporarily makes room for prompt
encoding when required by the device capacity.

#### Environment

- Platform: NVIDIA CUDA
- Workers: 4
- Default tensor parallel size: 1
- Default Ulysses degree / sequence parallel size: 4
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

#### Cache-DiT acceleration

MAGI-2 supports the shared Cache-DiT backend. Add these flags to either shared
offline command:

```text
--cache-backend cache_dit \
--enable-cache-dit-summary
```

The adapter wraps only `transformer.block.layers`, the repeated native
denoising stack. The pre-adapter, post-adapter, packed CFG preparation,
learned-sink attention, and video/audio decoders still execute normally.
MAGI-2 packs conditional and unconditional tokens into one transformer call,
so Cache-DiT treats each call as one denoising step rather than alternating CFG
passes. The same adapter composes with the DLO layouts below; DLO continues to
own layer placement while Cache-DiT decides whether the middle layers need to
execute for a step.

Cache-DiT is approximate acceleration. Keep the resident SP4 command without
`--cache-backend` as the reference-aligned quality baseline, and validate the
selected cache policy against that baseline for quality-sensitive workloads.

The adapter lifecycle was exercised locally on four NVIDIA L20X GPUs with the
released checkpoint, resident SP4, 272p, and four denoising steps. All ranks
installed and refreshed Cache-DiT; the shared T2VA command completed with 125
448x256 frames at 12.5 fps plus stereo 44.1 kHz audio. Peak reserved HBM was
65.95 GiB per worker. The shared script's default policy uses four warmup steps,
so this short run validates integration and output contracts, not cache speedup
or quality. Focused three-layer checks force a cache hit under resident,
HSDP4+SP4, and HSDP4+CFG2xSP2 execution and verify on every rank that only the
configured front block reruns on the cached step.

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

#### Full-quality four-GPU DLO qualification

The full Preview workflow was exercised from source head `a5af2a8c` on four
NVIDIA L20X GPUs in deterministic BF16 mode. Both runs used rank-local DLO SP4
with all 40 transformer blocks streamed and zero resident layers:

```text
--tensor-parallel-size 1 \
--ulysses-degree 4 \
--enable-distributed-layerwise-offload \
--dlo-no-use-allgather \
--dlo-resident-layers 0
```

The shared T2VA command above completed all 100 inference steps at 540p in
563.855 seconds. The shared I2VA command completed the same 100-step/540p
contract in 556.533 seconds, using a native 896x512 frame as its image
condition. Both outputs contain 125 H.264 frames at 12.5 fps for 10 seconds and
stereo 44.1 kHz AAC audio. Their MP4 SHA-256 values are respectively
`3437d078e1ce2358e4d02554ce7cdf720b1a93bf71055f4a9b53de0e6f16fdea` and
`b8c2a4acd333c96c6dfbbdc2b417cc018c22e38daa349b7e23aefd0966ff2de8`.

These are single E2E qualification runs, not benchmark samples. The generated
artifacts remain outside the repository, and no model-specific example or
benchmark script is added.

#### Other native four-GPU resident layouts

The native transformer also supports tensor parallelism. All three resident
layouts below were exercised end to end on four NVIDIA L20X GPUs:

| Layout | Shared-example flags | Distribution |
|---|---|---|
| SP4 (default) | `--tensor-parallel-size 1 --ulysses-degree 4` | Ulysses token/head partition plus rank-local MoE-head shards |
| TP2SP2 | `--tensor-parallel-size 2 --ulysses-degree 2` | Native TP matrix shards and row reductions inside each two-rank TP group, plus two-way Ulysses |
| TP4 | `--tensor-parallel-size 4 --ulysses-degree 1` | Native four-way TP matrix shards and row reductions |

Use the same shared T2VA or I2VA command and replace only the two parallelism
flags. The pipeline requires `TP x SP = 4` for a resident four-worker run and
validates attention-head, MoE-head, hidden-size, and intermediate-size
divisibility. The TP and SP layouts are deterministic within a fixed topology,
but BF16 collective order can produce topology-dependent numeric differences;
use SP4 when fidelity against the reference-aligned baseline is the priority.

#### HSDP, CFG parallelism, and distributed TurboVAE decode

The native pipeline also composes the shared HSDP, two-branch CFG, and VAE
parallel primitives with MAGI-2's Ulysses path. Add the relevant flags to the
shared T2VA or I2VA command:

| Four-worker layout | Additional flags |
|---|---|
| HSDP4 + SP4 | `--ulysses-degree 4 --use-hsdp --hsdp-shard-size 4` |
| CFG2 x SP2 | `--ulysses-degree 2 --cfg-parallel-size 2` |
| HSDP4 + CFG2 x SP2 | `--ulysses-degree 2 --cfg-parallel-size 2 --use-hsdp --hsdp-shard-size 4` |
| TurboVAE tile decode across four workers | `--vae-patch-parallel-size 4 --vae-use-tiling` |

HSDP uses FSDP2 for the dense transformer parameters while preserving MAGI's
already-SP-sharded MoE parameters. It cannot be combined with TP or DLO. CFG
parallelism assigns the positive and negative branches to separate CFG ranks
and preserves MAGI-2's distinct video/audio guidance, dynamic CFG, skimming,
and rescaling rules. `CFG2 x SP4` requires eight workers; use `CFG2 x SP2` on a
four-worker host.

Local qualification covered the real four-rank FSDP2 + SP4 collective path and
the combined HSDP4 + CFG2 x SP2 path with small native transformers. Because
one of the four GPUs was occupied by an unrelated live service, the released
checkpoint was exercised with HSDP3 + SP3 on the other three GPUs. Its
deterministic one-step 272p output was byte-identical to resident SP3, including
125 video frames and stereo audio. This SP3 run is bring-up evidence, not a
supported deployment recommendation.

TurboVAE patch parallelism distributes the decoder's exact temporal tile
chunks across the complete worker group; it does not introduce approximate
spatial crops. A released-checkpoint 540p PP4 decode matched resident decode
exactly (`max_abs_error=0`, identical SHA-256). Audio decode remains on the
output rank. VAE parallelism currently requires tile mode and a parallel size
equal to the complete DiT worker count.

#### Distributed layerwise offload

Distributed layerwise offload (DLO) streams the 40 Preview transformer blocks
from host memory. The following four-GPU layouts were exercised end to end:

| Layout | Parallelism and DLO flags | AllGather | Concurrent requests |
|---|---|:---:|:---:|
| DP4 | `--data-parallel-size 4 --tensor-parallel-size 1 --ulysses-degree 1 --enable-distributed-layerwise-offload --dlo-resident-layers 0` | Yes (default) | 4 |
| DP2SP2 | `--data-parallel-size 2 --tensor-parallel-size 1 --ulysses-degree 2 --enable-distributed-layerwise-offload --dlo-resident-layers 0` | Yes (default) | 2 |
| SP4 | `--tensor-parallel-size 1 --ulysses-degree 4 --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 0` | No, rank-local | 1 |

For DP4 and DP2SP2, DLO first applies the native local weight transform (such
as the SP-local MoE-head slice), then stores an orthogonal DP shard and
reconstructs the transformed tensor with AllGather for each block. All DP
ranks must therefore advance together. Send exactly `data_parallel_size`
concurrent requests with the same explicit `num_inference_steps` value.

SP4 ranks own different MoE-head shards, so SP-only DLO must use rank-local
streaming with `--dlo-no-use-allgather`. Add these flags to either shared
offline command to use that profile:

```text
--enable-distributed-layerwise-offload \
--dlo-no-use-allgather \
--dlo-resident-layers 0
```

The deterministic decoded video and audio from the SP4 DLO profile matched the
resident SP4 run exactly. The shared offline entrypoints expose TP, SP, and DLO
options, but not DP request-wave configuration; use online serving for the DP4
and DP2SP2 AllGather profiles.

DLO with AllGather does not support tensor parallelism. Data parallelism is
supported only with DLO, and SP-only DLO with AllGather is rejected during
pipeline validation.

#### Eight-GPU configuration status

The topology validator also accepts compatible eight-worker factorizations.
Examples include resident SP8, TP2SP4, and TP4SP2; DLO DP8, DP4SP2, and DP2SP4
use AllGather, while SP8 uses rank-local no-AllGather streaming. These profiles
must still satisfy the same dimension checks and DLO restrictions above.

Only the four-GPU layouts were exercised locally for this integration. Treat
the eight-GPU profiles as supported configuration validation, not as local
runtime qualification.

## Online serving

This four-worker example uses the default resident SP4 topology:

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

To serve DLO DP4, replace the default parallelism flags with
`--data-parallel-size 4 --tensor-parallel-size 1 --ulysses-degree 1` and add
`--enable-distributed-layerwise-offload --dlo-resident-layers 0`. For DLO
DP2SP2, use `--data-parallel-size 2`, `--tensor-parallel-size 1`, and
`--ulysses-degree 2` with the same DLO flags. Submit four or two concurrent
requests, respectively, and keep `num_inference_steps` identical across the
request wave.

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

- Each replica processes one request at a time. DLO DP4 and DP2SP2 can process
  four and two matched requests concurrently across replicas; fused per-rank
  request batching is not supported.
- Only the published 10-second, 125-frame, 12.5-fps Preview workflow is supported.
- Native generation is limited to the `272p` and `540p` tiers.
- The worker world must contain 4 or 8 GPUs. Parallel dimensions must cover the
  full worker world and pass the model-dimension divisibility checks.
- Data parallelism requires DLO. DLO data-parallel replicas require TP=1, and
  DLO AllGather requires DP greater than 1.
- HSDP, cache acceleration, quantization, generic module CPU offload, CFG
  parallelism, ring sequence parallelism, pipeline parallelism, and VAE patch
  parallelism are not supported for this pipeline.
- SP-only DLO requires `--dlo-no-use-allgather`; SP ranks own different
  MoE-head shards and cannot form a DLO AllGather group with one another.
- Full 100-step 540p T2VA and I2VA were qualified only with four-device
  rank-local DLO SP4. The DP4/DP2SP2 DLO profiles retain bounded one-step and
  four-step coverage rather than full-quality qualification.
- The model card's official eight-Hopper runtime was not available on the
  four-L20X qualification host. Compatible eight-worker topologies pass
  configuration validation but remain runtime-unqualified in this PR.
