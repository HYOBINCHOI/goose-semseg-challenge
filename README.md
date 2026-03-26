# goose-semseg-challenge

GOOSE challenge repository for semantic segmentation.

# GOOSE Dataset: Image Processing

## Set-up

<details>

<summary><b>General requirements (Tested)</b></summary>

- Python 3.9.
- torch = 1.13.1
  - <https://pytorch.org/get-started/locally/>
- The python packages specified in `config/requirements.txt`

</details>

<br>

<details>
<summary><b>Conda automatic Set-up</b></summary>

We recomend using a [conda environment](https://docs.anaconda.com/miniconda/miniconda-install/):

```bash
source setup.sh
```

This will install and activate a conda environment with the necessary dependencies.

</details>

### 2D Semantic Training

**Example usage of `semantic_train.py`**

```bash
python semantic_train.py /path/to/goosedataset --epochs 100 --batch_size 64 --resize_width 512 --resize_height 512
```

### 2D Semantic Evaluation

To evaluate the performance of a trained checkpoint the script `evaluation.py` can be used.

**Example usage of `evaluation.py`**

```bash
python evaluation.py /path/to/goose-dataset /path/to/checkpoint.pt  --test_split_name val --train_script_dir /path/to/training/scripts
```

The results will be printed to the console and saved as a file to the output directory

### Model inference

To run the images through the models and save the inferred results use the `compare_gt.py` script.

**Example usage of `compare_gt.py`**

```bash
python compare_gt.py --checkpoint /path/to/checkpoint.pt --data_path /path/to/goose-dataset --train_script_dir /path/to/training/scripts --colormap /path/to/goose_colormap.json --split val --num_samples 10
```

The results will be saved to the output directory

## Pre-commit

We use [pre-commit](https://pre-commit.com/) to maintain consistent code style across the repository.

### Installation

```bash
pip install pre-commit
pre-commit install
```

### Usage

Once installed, pre-commit hooks will run automatically on `git commit`. To manually run all hooks on all files:

```bash
pre-commit run --all-files
```
