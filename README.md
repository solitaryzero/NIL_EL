# NIL_EL

This repository maintains code, data and for ACL Findings 2023 paper 

[Learn to Not Link: Exploring \nil{} Prediction in Entity Linking](https://arxiv.org/abs/2305.15725)

## Get Started

### Environment
```bash
git clone https://github.com/solitaryzero/NIL_EL.git
cd NIL_EL/src
pip install -r requirements.txt
```

### Build the NEL Dataset
The `scripts/preprocess` folder contains shell scripts for building the NEL dataset.

`vectorize.sh` builds the standard NEL dataset,
`vectorize_no_nil.sh` builds the dataset without NIL entries, 
`vectorize_partial.sh` and `vectorize_partial_nep.sh` build the dataset with only a portion of NIL or NEP entries.

### Training and Evaluation
In the `scripts/nel`, `scripts/nel_nonil`, `scripts/nel_partial` and `scripts/nel_partial_nep` folders, `train_[model].sh` works for training and `evaluate_[model].sh` works for evaluation.

The `model` names refer to:
- blink: the standard bi-encoder model
- clink: bi-encoder with type prediction
- cross: the standard cross-encoder model
- cross_clink: cross-encoder with type prediction

### Evaluation on Standard Datasets
The `scripts/standard` folder contains the evaluation scripts for standard datasets like AIDA, MSNBC and WNED.

The standard dataset benchmarks can be downloaded at [[Google Drive](https://drive.google.com/file/d/1JasibUxRGMbumKY4ZO7zLRz0yPhlOpfN/view?usp=sharing)]