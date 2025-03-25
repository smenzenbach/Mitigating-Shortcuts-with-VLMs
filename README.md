# Mitigating-Shortcuts-with-VLMs

This repository contains code to produce data following the pipeline described in the paper ....., as well as code to train and evaluate the models with the produced data.

## Pipeline
The pipeline is composed of three steps:
1. Classification of the shortcuts (is a confounder present or not)
2. Open-vocabulary detection and segmentation of the confounders
3. Inpainting of the confounders at segmentation masks of step 2

## How to use the code
First, clone the repository and go to the root directory of the project:
```bash
git clone ...
cd Mitigating-Shortcuts-with-VLMs
```

To run the code the repo has to be added to the PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/Mitigating-Shortcuts-with-VLMs
```

### Data
We have used the pipeline to generate data for the following datasets:
- ISIC 2019
- CelebA
- coco

Important information needed to run the pipeline, training and evaluation of the models is saved in a YAML file saved in the `data` directory. The YAML file contains the following information per dataset:

- `gt_label_path`: Path to the ground truth labels csv file
- `wr_masks_dir`: Path to the directory containing the segmentation masks for the wrong reason (confounded area)
- `rr_masks_dir`: Path to the directory containing the segmentation masks for the right reason (target area)
- `raw_dir`: Path to the directory containing the raw images
- `worst_group_class_label`: Class label of the worst group in the dataset


### Config file
To configure the models, prompts and other parameters used for the pipeline, a config file is used. The config file is a toml file that contains the following sections:

