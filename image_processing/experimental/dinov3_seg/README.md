# Experimental DINOv3 Segmentation

This package keeps the existing `image_processing` training and evaluation code
untouched while adding an isolated DINOv3 segmentation stack under
`experimental/dinov3_seg`.

## Layout

- `vendor/`: vendored upstream-style DINOv3 segmentation components
- `goose/`: thin GOOSE-specific wrappers and checkpoint helpers
- `train.py`: experimental training entrypoint
- `eval.py`: experimental evaluation entrypoint
- `config/train.yaml`: default config for the experimental trainer

## Reuse Strategy

- Existing `image_processing/scripts/*` helpers are reused by import only.
- Existing `image_processing/goosetools/*` dataset code is reused by import only.
- Existing `image_processing/models/*`, `scripts/*`, and `tools/*` files are not modified.

## Notes

- The vendored deformable-attention utility includes a PyTorch fallback path so
  this package can run without compiling the upstream CUDA extension. This is
  slower than the compiled path but keeps the package self-contained.
- The training loop reuses the current project checkpoint, metric, dataloader,
  and distributed helpers.

## Example

Train:

```bash
cd /home/mipstu/jiPark/challenge/goose-semseg-challenge/image_processing

python experimental/dinov3_seg/train.py \
  --config experimental/dinov3_seg/config/train.yaml
```

Evaluate:

```bash
cd /home/mipstu/jiPark/challenge/goose-semseg-challenge/image_processing

python experimental/dinov3_seg/eval.py \
  /home/datasets/goose-dataset \
  /path/to/checkpoint.pt \
  --label_mapping_csv /home/datasets/goose-dataset/goose_label_mapping.csv
```
