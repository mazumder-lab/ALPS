# ALPS: Improved Optimization for Highly Sparse One-Shot Pruning for Large Language Models

This is the offical repo of the NeurIPS 2024 paper **ALPS: Improved Optimization for Highly Sparse One-Shot Pruning for Large Language Models**


## Requirements: 
This code has been tested with Python 3.9.16 and the following packages:
+ torch 2.0.0
+ transformers 4.35.2
+ datasets 2.15.0
+ numpy 1.24.3

## Datasets and models

The data files can be found at https://www.dropbox.com/scl/fi/6xg1voa7go9x2uds1y2mq/scd_data.zip?rlkey=8bwzshamiyvcvpd146ymkp0vc&dl=0.

To download the model, set cached to False in `get_opt` and `get_llama` and run the script. After the model is downloaded, upload the model files to the cluster and set cached to True. 


## Running:

+ Create two folders `results` and `pruned_models`

+ Run Python files:
    - python opt.py facebook/opt-125m c4 ALPS {sparsity} --model_path {your_path} --data_path {your_path}

     - python llama.py meta-llama/Llama-2-7b-hf c4 ALPS {sparsity} --model_path {your_path} --data_path {your_path}

+ We usually use c4 as the training (calibration) data. For additional configuration options, refer to `opt.py` and `llama.py`. The results of perplexity and zero-shot evaluations will be stored in the `results` directory, while the pruned models will be saved in the `pruned_models` directory.


## Citing ALPS:

If you find ALPS useful in your research, please consider citing the following paper.

```
@article{meng2024alps,
  title={ALPS: Improved Optimization for Highly Sparse One-Shot Pruning for Large Language Models},
  author={Meng, Xiang and Behdin, Kayhan and Wang, Haoyue and Mazumder, Rahul},
  journal={arXiv preprint arXiv:2406.07831},
  year={2024}
}
```
