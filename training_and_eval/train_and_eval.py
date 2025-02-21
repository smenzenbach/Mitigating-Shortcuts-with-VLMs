import argparse
import datetime
from setup_data import Dataset
from learner.models import VGG16_pretrained
from utils import seed_all
import torch.nn as nn
import logging
import torch
import os

from learner.learner import Learner

# python train_and_eval.py --dataset ISIC19 -d 1 --id eval_test_vanilla

parser = argparse.ArgumentParser(
    description="SHORTCUT MITIGATION THROUGH INPAINTING EVAL"
)

parser.add_argument(
    "-ds",
    "--dataset",
    choices=["ISIC19", "CelebA"],
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
    "--model", default="vgg16", choices=["vgg16"], help="Which model should be used?"
)  # add more models
parser.add_argument(
    "--id",
    default=None,
    help="ID to identify trained model. If None then timestamp is used.",
)
args = parser.parse_args()

DEVICE = f"cuda:{args.device}"
LR = 0.001
SEED = 1
BATCH_SIZE = 16
TRAIN_SHUFFLE = True
SAVE_LAST = True
SCHEDULER = True
EPOCHS = 50
VERBOSE_AFTER_N_EPOCHS = 1

print(f"Using device {DEVICE} and seed {SEED}")

# get dir of repo (parent dir)
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load data
dataset = Dataset(dataset_name=args.dataset, model_name=args.model)

dataloaders, loss_weights = dataset.load_dataloaders_and_loss_weights(
    augmented_images_dir=args.augmented_images_dir,
    generated_masks_dir=args.generated_masks_dir,
    batch_size=BATCH_SIZE,
)

train_dataloader, val_loader, test_dataloader = (
    dataloaders["train"],
    dataloaders["val"],
    dataloaders["test"],
)

########### Initalize model, loss and optimizer
# Logging
logging.basicConfig(
    filename=f"{repo_dir}/training_and_eval/shortcut_mitigation_training.log",
    level=logging.INFO,
    filemode="a",
    format="%(asctime)s %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
)
logging.info(
    f"\n\n--------RPEXP Train {args.model}, base_criterion=CEL, dataset={args.dataset}--------"
)
logging.info(
    f"-HYPERPARAMS epochs= {EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, optim=SGD(momentum=0.9), save_last={SAVE_LAST}, seeds={SEED}, scheduler={SCHEDULER}, shuffle={TRAIN_SHUFFLE}"
)

# Set seed
seed_all(SEED)

# Load model
if args.id is not None:
    identifier = args.id
else:
    identifier = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

MODELNAME = f"{dataset.dataset_name}-{identifier}"

if args.model == "vgg16":
    model = VGG16_pretrained(freeze_features=False).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

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
)

# fit
if not args.load:
    learner.fit(
        train_dataloader,
        val_loader,
        epochs=EPOCHS,
        save_best=False,
        save_last=SAVE_LAST,
        verbose=True,
        verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS,
    )

########### Evaluate
# Accuracy
learner.validation_statistics(test_dataloader)
logging.info("Accuracy on test set DONE.")

# WR
learner.get_wr_metric(test_dataloader)
logging.info("WR on test set DONE.")

# RR
learner.get_rr_metric(test_dataloader)
logging.info("RR on test set DONE.")
