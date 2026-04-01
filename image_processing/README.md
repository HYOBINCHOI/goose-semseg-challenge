### Model architecture

- Built a ConvNeXt + Mask2Former segmentation model for the GOOSE dataset.
- Used a ConvNeXt-based backbone together with a Mask2Former segmentation head.
- Configured the model for the project setting with 64 semantic classes.
- Kept the overall pipeline focused on semantic segmentation training and evaluation for the GOOSE challenge.

______________________________________________________________________

### Model

The model code is organized into two main files.

- `models/model.py`

  - project-specific wrapper for ConvNeXt + Mask2Former
  - handles model construction and dataset-specific configuration
    <br>

- `models/mask2former_model.py`

  - local Mask2Former implementation
  - contains the core architecture and loss logic

______________________________________________________________________

### Training code refactor

The original training script was split into smaller modules for readability and maintenance.

- `scripts/args.py`

  - config loading
  - argument parsing
  - config validation

- `scripts/checkpoint_utils.py`

  - checkpoint save/load helpers
  - epoch logging
  - training curve export

- `scripts/dataset_utils.py`

  - collator
  - batch/device utilities
  - semantic map to mask/class target conversion

- `scripts/metrics.py`

  - semantic prediction conversion
  - confusion matrix
  - mIoU computation

- `scripts/train.py`

  - training entrypoint
  - main training flow

- `scripts/train_utils.py`

  - training loop
  - epoch execution
  - optimizer setup
  - resume logic

- `scripts/utils.py`

  - seed setup
  - device and distributed setup
  - dataset path utilities
  - output directory generation

______________________________________________________________________
### Submission file generation

Use `tools/make_submission_zip.py` to generate prediction PNGs and package them into a submission zip for the website.

#### Example

```bash
python /goose-semseg-challenge/image_processing/tools/make_submission_zip.py \
  --dataset_root /home/datasets/goose-dataset \
  --checkpoint /path/to/model.pt \
  --output_dir /path/to/submission_pngs \
  --output_zip /path/to/submission.zip
```
______________________________________________________________________
### Notes
