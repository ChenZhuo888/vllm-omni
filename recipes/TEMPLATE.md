# Recipe Title

> Example: Qwen3-Omni for speech chat

## Summary

- Vendor:
- Model:
- Task:
- Mode:
- Maintainer:

## When to use this recipe

Briefly describe the concrete scenario this recipe covers.

## Supported model contract

Put user-visible constraints before commands. Use compact tables for:

- supported tasks and shared entrypoints;
- input modalities, counts, duration/geometry limits, and required combinations;
- output duration, geometry, frame/audio rates, and variants;
- the hardware/deployment profiles that this recipe actually provides.

Source model specifications from the canonical model card or repository. Do
not copy values from a sibling model or infer support from configuration alone.

## References

- Upstream or canonical docs:
- Related example under `examples/`:
- Related issue or discussion:

## Hardware Support

Add one section per platform, such as `GPU`, `ROCm`, or `NPU`. Under each
platform section, document one or more tested hardware configurations. Name
the accelerator model, per-device memory, device count, device interconnect,
and relevant host-memory capacity. Separate upstream requirements and
configuration-only support from locally runtime-qualified hardware.

## GPU

### 1x A100 80GB

#### Hardware

- Accelerator model and per-device memory:
- Number of devices:
- Device interconnect (NVLink, PCIe, or other):
- Host memory, when CPU offload or staging is relevant:

#### Software environment

- OS:
- Python:
- Driver / runtime:
- vLLM version:
- vLLM-Omni version or commit:

Keep worker counts, precision, and TP/SP/DP/PP sizes with the command or
deployment profile; they are not software-environment properties.

#### Command

```bash
# Add the exact command(s) here
```

#### Verification

```bash
# Add a quick validation command or expected output here
```

#### Notes

- Memory usage:
- Key flags:
- Known limitations:

### 2x L40S

Repeat the same structure for other hardware setups as needed.

## ROCm

### Example hardware configuration

Repeat the same nested structure for ROCm setups as needed:

- `#### Hardware`
- `#### Software environment`
- `#### Command`
- `#### Verification`
- `#### Notes`

## NPU

### Example hardware configuration

Repeat the same nested structure for NPU setups as needed:

- `#### Hardware`
- `#### Software environment`
- `#### Command`
- `#### Verification`
- `#### Notes`

## Supported features

Use a compact table containing the model-specific status/topology and a link to
the corresponding shared feature guide. Do not repeat the general feature
documentation in the recipe. Record unsupported combinations in the same table
instead of a disconnected restrictions section.
