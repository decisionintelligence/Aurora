# <img src="doc/logo.png" alt="Image description" style="width:55px;height:55px;"> Aurora: Towards Universal Generative Multimodal Time Series Forecasting

**The official code repo of our ICLR 26's paper: [Aurora: Towards Universal Generative Multimodal Time Series Forecasting](https://arxiv.org/pdf/2509.22295)**

[![ICLR](https://img.shields.io/badge/ICLR'26-Aurora-orange)](https://arxiv.org/pdf/2509.22295)  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)  [![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-blue)](https://pytorch.org/)  ![Stars](https://img.shields.io/github/stars/decisionintelligence/Aurora) 



## Introduction

Aurora is a highly capable multimodal time series foundation model. Based on the **Modality-Guided Multi-head Self-Attention** and  **Prototype-Guided Flow Matching**, Aurora can effectively utilize the domain-specific knowledge contained in modalities and support generative probabilistic forecasting, thus covering versatile forecasting scenarios.

See **Figure 1**, to our best knowldege, Aurora is the first pretrained multimodal time series foundation model! Evaluated on three well-recognized benchmarks, including TimeMMD, TSFM-Bench, and ProbTS, Aurora is demonstrated the state-of-the-art.

<div align="center">
<img alt="intro" src="doc/intro.png" width="50%"/>
</div>

## Architecture

In this work, we pretrain Aurora in a cross-modality paradigm, which adopts Channel-Independence on time series data, and models corresponding multimodal interaction to inject domain knowledge. Note that the each variable of time series is first normalized through Instance Normalization to mitigate the value discrepancy. See **Figure 2**, Aurora mainly consists of two phases: 1) in Aurora Encoder, we tokenize and encode each modality into modal features, then fuse them to form multimodal representations; 2) in Aurora Decoder, we utilize a Condition Decoder to obtain the multimodal conditions of future tokens, leverage a Prototype Retreiver to retrieve the future prototypes based on the domain knowledge, and conduct flow matching on them to make generative probabilistic forecasts.

<div align="center">
<img alt="arch" src="doc/arch.png" width="80%"/>
</div>
## Quickstart

#### From pypi (recommended)

We have publised Aurora on PyPi, **you can directly install it with one line of code!**

```shell
$ pip install aurora-model==0.1.0
```

Then you can use the Aurora model to make zero-shot probabilistic forecasting!

```python
from aurora import load_model
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model = load_model()

# prepare input
batch_size, lookback_length = 1, 528 
seqs = torch.randn(batch_size, lookback_length)

# Note that Sundial can generate multiple probable predictions
forecast_length = 96 
num_samples = 100


# For inference_token_len, you can refer to LightGTS (Periodic Patching).
# We recommend to use the period length as the inference_token_len.
output = model.generate(inputs=seqs, max_output_length=forecast_length, num_samples=num_samples, inference_token_len=48)


# use raw predictions for mean/quantiles/confidence-interval estimation
print(output.shape) 
```



#### From raw code

We release the original code of Aurora in this repo. You can also download the pretrained checkpoints in our [huggingface](https://huggingface.co/DecisionIntelligence/Aurora) repo and put them in the folder: aurora/.

If you want to pretrain an Aurora on your own time series corpus, you need to install the following important packages:

```shell
$ pip install torch==2.4.0
$ pip install torchvision==0.19.0
$ pip install transformers[torch]
```

```python
from huggingface_hub import snapshot_download
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- Configuration ---
repo_id = "DecisionIntelligence/Aurora" 

# Target directory for the download. "." represents the current working directory.
local_dir = "./work_dir" 

# Optional: Set repo_type to "dataset" or "space" if you are not downloading a model.
repository_type = "model" 
# ---------------------

# Ensure the local directory exists before starting the download
if not os.path.exists(local_dir):
    os.makedirs(local_dir)
    print(f"Created directory: {local_dir}")

print(f"Starting download from '{repo_id}' to '{local_dir}'...")

try:
    # snapshot_download handles the download of all files in the repository
    download_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        # Set to False to download actual files instead of symbolic links
        local_dir_use_symlinks=False, 
        repo_type=repository_type,
        # Use your HF access token for private/gated repositories
        token=None 
    )
    print(f"\nSuccess! All files downloaded to: {download_path}")

except Exception as e:
    print(f"\nAn error occurred during download: {e}")

# Then you can easily make zero-shot forecasts using Aurora

from modeling_aurora import AuroraForPrediction

model = AuroraForPrediction.from_pretrained("./",trust_remote_code=True)

# prepare input
batch_size, lookback_length = 1, 528 
seqs = torch.randn(batch_size, lookback_length)

# Note that Sundial can generate multiple probable predictions
forecast_length = 96 
num_samples = 100


# For inference_token_len, you can refer to LightGTS (Periodic Patching).
# We recommend to use the period length as the inference_token_len.
output = model.generate(inputs=seqs, max_output_length=forecast_length, num_samples=num_samples, inference_token_len=48)


# use raw predictions for mean/quantiles/confidence-interval estimation
print(output.shape) 
```



## Experiments

For benchmarking (TSFM-Bench, ProbTS, TimeMMD, TFB, and EPF), you can install additional packages based on the requirement files under folders, and the datasets can be fetched from this [link](https://drive.google.com/file/d/12tJk858WaoG7ZVSvUq8KU1oHfGNJrARF/view?usp=drive_link). All experimental results can be reproduced by running the scripts in the benchmark folder：

```shell
# TimeMMD
TimeMMD/scripts/run_aurora_timemmd_zero_shot.sh

# EPF
EPF/scripts/run_aurora_short_term_zero_shot.sh

# ProbTS
ProbTS/scripts/run_aurora_probts.sh

# TSFM-Bench
TFB/scripts/run_aurora_tfb.sh

# TFB univaraite
TFB/scripts/run_aurora_uni.sh
```



## Performance

**Aurora ahieves consistent state-of-the-art performance on these 5 benchmarks:**

<div align="center">
<img alt="arch" src="doc/table1.png" width="1000%"/>
</div>

<div align="center">
<img alt="arch" src="doc/table2.png" width="100%"/>
</div>

<div align="center">
<img alt="arch" src="doc/table3.png" width="100%"/>
</div>

<div align="center">
<img alt="arch" src="doc/table4.png" width="100%"/>
</div>

<div align="center">
<img alt="arch" src="doc/table5.png" width="100%"/>
</div>



## Citation

If you find this repo useful, please cite our paper.

```latex
@inproceedings{wu2026aurora,
  title     = {Aurora: Towards Universal Generative Multimodal Time Series Forecasting},
  author    = {Wu, Xingjian and Jin, Jianxin and Qiu, Wanghui and Chen, Peng and Shu, Yang and Yang, Bin and Guo, Chenjuan},
  booktitle = {ICLR},
  year      = {2026}
}
```



## Contact

If you have any questions or suggestions, feel free to contact:

- [Xingjian Wu](https://ccloud0525.github.io/) ([xjwu@stu.ecnu.edu.cn](mailto:xjwu@stu.ecnu.edu.cn))
- [Peng Chen](https://pengchen12.github.io/) (pchen@stu.ecnu.edu.cn)

Or describe it in Issues.
