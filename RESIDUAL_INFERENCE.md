# Residual Text-Condition Streaming for SD3 / SD3.5

This repository now supports an optional **residual condition streaming** path that
lets you inject previously computed text tokens (the "text streaming tokens" used
by MMDiT) into deeper layers of the transformer during inference. The feature is
available for both SD3 and SD3.5 checkpoints and can be controlled entirely from
`sd3_infer.py`.

## How the residual hooks work

- **`residual_origin_layer`** &mdash; zero-indexed layer id that provides the
  tokens to be re-used. `0` refers to the embeddings *before* the first MMDiT
  block receives them. Any other value uses the tokens that are about to enter
  the specified block.
- **`residual_target_layers`** &mdash; a list of block indices that should receive
  the residual addition. Each listed block will have the origin tokens mixed into
  its condition stream right before the block executes.
- **`residual_weights`** &mdash; weights applied to the origin tokens before being
  added into every target block. The list length must match the target layer
  list; when omitted, every target defaults to weight `1.0`.

For example, running with `--residual_origin_layer=0 --residual_target_layers=3,5,7`
`--residual_weights=0.8,0.6,0.4` mixes the original text tokens (before any
block) into blocks 3, 5 and 7 with progressively smaller weights.

## CLI usage

All options are available directly through `sd3_infer.py`:

```bash
python sd3_infer.py \
  --prompt "neon street at night" \
  --model models/sd3.5_large.safetensors \
  --vae models/sd3_vae.safetensors \
  --residual_origin_layer 0 \
  --residual_target_layers 3,5,7 \
  --residual_weights 0.8,0.6,0.4
```

Both comma-separated and whitespace-separated lists are accepted, so
`--residual_target_layers "3 5 7"` also works. Leave the target list empty to
disable the feature without changing other arguments.

## Convenience launcher

A small helper script, [`run_inference.sh`](run_inference.sh), wraps the
arguments needed for SD3 and SD3.5:

```bash
./run_inference.sh sd3.5 "cinematic portrait" \
  --residual_origin_layer 0 \
  --residual_target_layers 4,8,12 \
  --residual_weights 1.0,0.5,0.25
```

The script assumes all weights live in `./models` but you can override the
paths via environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MODEL_DIR` | Directory containing the weight files | `models` |
| `SD3_MODEL_PATH` | Override SD3 checkpoint location | `$MODEL_DIR/sd3_medium.safetensors` |
| `SD35_MODEL_PATH` | Override SD3.5 checkpoint location | `$MODEL_DIR/sd3.5_large.safetensors` |
| `VAE_PATH` | Override VAE checkpoint | `$MODEL_DIR/sd3_vae.safetensors` |

Any additional arguments after the prompt are forwarded directly to
`sd3_infer.py`, so you can keep using existing flags (e.g. `--steps`, `--cfg`,
`--controlnet_ckpt`, etc.).

## Output organization

`sd3_infer.py` still creates an output directory per prompt/model combination,
so enabling residual streaming does not change how images are saved. You can mix
and match conventional sampling options with the new residual controls without
any further code changes.
