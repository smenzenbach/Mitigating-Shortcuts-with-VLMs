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
To configure the models, prompts and other parameters used for the pipeline, a config file is used. The config file is a TOML file that contains the following sections:

- `dataset`: Contains the dataset name and the path to the YAML file containing the dataset information
    - `name`: Name of the dataset (same as directory name of the dataset; e.g. `CelebA`, `ISIC19` or `coco`)

- `confounder_classification`: Contains the parameters for the confounder classification model (Optional)
    - `model`: Name of the model/method to use (e.g. `sgl`)
    - `output_for_true`: Output value expected from model for the presence of a confounder, i.e. what answers are considered as true (confounder present)
    - `batch_size`: After how many images the results are saved to csv file
    - `resize_size`: Maximum pixel size of longest side of the image, keeps aspect ratio (Optional)

- `confounder_classification.sgl`: Contains the parameters for the SGL model (if `model` is set to `sgl`)
    - `user_input_list`: List of user inputs to use for the model. The number of items in the list determine the number QA turns.
    - `model_path`: path to Huggingface model
    - `tokenizer_path`: path to Huggingface tokenizer (Optional, Default: model_path)
    - `chat_template`: Chat template to use for the model (Optional)
   
- `segmentation`: Contains the parameters for the detection and segmentation model
    - `model`: Name of the model to use (e.g. `groundingRMBG`)
    - `custom_vocabulary`: List of words to use for the open-vocabulary detection
    - `box_lower_threshold`: Lower threshold for how much area of the image is allowed to be covered by the bounding box (Optional)
    - `box_upper_threshold`: Upper threshold for how much area of the image is allowed to be covered by the bounding box (Optional)
   
- `inpainting`: Contains the parameters for the inpainting model
    - `model`: Name of the model to use (e.g. `lama_sdxl`, `lama`, `sdxl`)
    - `prompts`: JSON of dict for each prompt. Each dict has the key `prompt` and optionally `filter_func` and `num_imgs`. `filter_func` is a function that filters the images . `num_imgs` is the number of images to use for the prompt 
    - `batch_size`: number of images to process in a batch (Optional,Default: 50)
    - `dilation_iterations`: Number of dilation iterations to apply to the segmentation mask (Optional,Default: 5)

### Run confounder classification
Depending on the model used you need to login to the Huggingface hub to have access. To run the confounder classification step, use the following command:
```bash
python pipeline/pipe_confounder_classification/run_confounder_classification.py --config /path/to/config.toml --device gpu_id
```
`--device` or short `-d` is the id of the GPU to use. If not provided, the code will use 0 as default.

### Run segmentation
To run the segmentation step, use the following command:
```bash
python pipeline/pipe_segmentation/run_segmentation.py --config /path/to/config.toml --device gpu_id
```

### Run inpainting
To run the inpainting step, use the following command:
```bash
python pipeline/pipe_inpainting/run_inpainting.py --config /path/to/config.toml --device gpu_id
```

### Training and evaluation
To train and evaluate the models, use the following command:
```bash
python training_and_eval/train_and_eval.py --dataset dataset_name -d gpu_id --id identifier_of_run --model model_name --batch_size 64 --augmented_images_dir /path/to/inpainting
```
- `dataset_name`: Name of the dataset (e.g. `CelebA`, `ISIC19` or `coco`)
- `gpu_id`: ID of the GPU to use, default is 0
- `identifier_of_run`: Identifier of the run
- `model_name`: Name of the model to use (e.g. `resnet50`, `vgg16`)
- `batch_size`: Batch size to use for training
- `augmented_images_dir`: Path to the directory containing the inpainted images

### Results
The results of the models are saved in the `training_and_eval/eval_output` directory. For evaluation we save the accurcy, the worst group accuracy, and the wrong reason and right readon metric.




