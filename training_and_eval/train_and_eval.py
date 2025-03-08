import argparse
import datetime

from setup_data import Dataset
from learner.models import VGG16_pretrained, resnet50_pretrained
from utils import seed_all, MyLogger
import torch.nn as nn
import logging
import torch
import os

from learner.learner import Learner


# python train_and_eval.py --dataset ISIC19 -d 0 --id vanilla
# python train_and_eval.py --dataset ISIC19 --augmented_images_dir /workspace/home/pipeline_outputs/ISIC19/pipeline_test/inpainting -d 2 --id pipeline -l
# python train_and_eval.py --dataset CelebA -d 1 --id vanilla_test
# python train_and_eval.py --dataset CelebA -d 3 --id vanilla -l
# python train_and_eval.py --dataset CelebA --augmented_images_dir /workspace/home/pipeline_outputs/CelebA/lama_sdxl/inpainting -d 2 --id lama_sdxl -l --model vgg16
# python train_and_eval.py --dataset coco -d 3 --id vanilla --model resnet50 --batch_size 64
# python train_and_eval.py --dataset coco -d 3 --id lama_sdxl --model resnet50 --batch_size 64 --augmented_images_dir /workspace/home/pipeline_outputs/coco/racket_person/inpainting

parser = argparse.ArgumentParser(
    description="SHORTCUT MITIGATION THROUGH INPAINTING EVAL"
)

parser.add_argument(
    "-ds",
    "--dataset",
    # choices=["ISIC19", "CelebA", "coco"],
    required=True,
    help="Which dataset should be used?",
)

parser.add_argument(
    "--augmented_images_dir",
    default=None,
    help="Path to augmented images that should be used for training.",
)
parser.add_argument(
    "--generated_masks_dir",
    default=None,
    help="Path to generated masks that are used for loss augmentation mode.",
)

parser.add_argument(
    "-d",
    "--device",
    default=0,
    type=int,
    help="Which GPU to use? Please give the id of the GPU",
)
parser.add_argument("-l", "--load", action="store_true", help="If to load model")

parser.add_argument(
    "-m",
    "--model",
    default="resnet50",
    choices=["vgg16", "resnet50"],
    help="Which model should be used?",
)  # add more models

parser.add_argument(
    "--id",
    default=None,
    help="ID to identify trained model. If None then timestamp is used.",
)

parser.add_argument("-bs", "--batch_size", default=16, type=int, help="Batch size")

args = parser.parse_args()

DEVICE = f"cuda:{args.device}"
LR = 0.001
SEED = 1
BATCH_SIZE = args.batch_size
TRAIN_SHUFFLE = True
SAVE_LAST = True
SCHEDULER = True
EPOCHS = 50
VERBOSE_AFTER_N_EPOCHS = 1

# get dir of repo (parent dir)
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# setup general logger over all runs
logger_dir = f"{repo_dir}/training_and_eval/logs"
general_logger = MyLogger(f"{logger_dir}/shortcut_mitigation_training.log").logger


# get identifier for run
if args.id is not None:
    identifier = args.id
else:
    identifier = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# set up identifier specific logger
os.makedirs(logger_dir, exist_ok=True)
logger = MyLogger(f"{logger_dir}/{args.dataset}-{identifier}.log").logger

general_logger.info(f"Load data for {identifier}")
logger.info(f"Loading data....")


# Load data
dataset = Dataset(dataset_name=args.dataset, model_name=args.model, logger=logger)

dataloaders, loss_weights = dataset.load_dataloaders_and_loss_weights(
    identifier=identifier,
    augmented_images_dir=args.augmented_images_dir,
    generated_masks_dir=args.generated_masks_dir,
    batch_size=BATCH_SIZE,
)

general_logger.info(f"Data loaded for {identifier}. \n")
logger.info(f"Data loaded.\n")

train_dataloader, val_dataloader, test_dataloader, wg_test_dataloader = (
    dataloaders["train"],
    dataloaders["val"],
    dataloaders["test"],
    dataloaders["worst_group_test"],
)

########### Initalize model, loss and optimizer
if args.model == "vgg16":
    model = VGG16_pretrained(freeze_features=False).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    opt_str = "SGD(momentum=0.9)"
elif args.model == "resnet50":
    model = resnet50_pretrained(freeze_features=False).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    opt_str = "Adam"

# Logging
# training info for general logger
general_logger.info(
    f"--------RPEXP Train {args.model}, base_criterion=CEL, dataset={args.dataset}, identifier={identifier}--------"
)
general_logger.info(
    f"-HYPERPARAMS epochs= {EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, optim={opt_str}, save_last={SAVE_LAST}, seeds={SEED}, scheduler={SCHEDULER}, shuffle={TRAIN_SHUFFLE}\n"
)

# training info for specific logger
logger.info(
    f"--------RPEXP Train {args.model}, base_criterion=CEL, dataset={args.dataset}, identifier={identifier}--------"
)
logger.info(
    f"-HYPERPARAMS epochs= {EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, optim={opt_str}, save_last={SAVE_LAST}, seeds={SEED}, scheduler={SCHEDULER}, shuffle={TRAIN_SHUFFLE}\n"
)

# Set seed
seed_all(SEED)

# Load model
MODELNAME = f"{dataset.dataset_name}-{identifier}"


# loss
base_criterion = nn.CrossEntropyLoss(weight=loss_weights["train"].to(DEVICE))
loss = base_criterion

# learner
learner = Learner(
    model,
    loss,
    optimizer,
    DEVICE,
    MODELNAME,
    base_criterion=base_criterion,
    load=args.load,
    logger=logger,
)

# fit
learner.fit(
    train_dataloader,
    val_dataloader,
    epochs=EPOCHS,
    save_best=False,
    save_last=SAVE_LAST,
    verbose=True,
    verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS,
    use_checkpoint=args.load,
)

########### Evaluate
# Accuracy
learner.validation_statistics(test_dataloader)
logger.info("Accuracy on test set DONE.")

# Worst-group accuracy
if wg_test_dataloader is not None:
    learner.validation_statistics(wg_test_dataloader, worst_group=True)
    logging.info("Accuracy on worst-group test set DONE.")
else:
    logging.info("No worst-group test set provided.")

# WR
learner.get_wr_metric(test_dataloader)
logger.info("WR on test set DONE.")

# RR
learner.get_rr_metric(test_dataloader)
logger.info("RR on test set DONE.\n")
