"""Learner class with different utility functions."""

import logging
import time
import uuid
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from rtpt import RTPT

from utils import explain_with_ig_one_by_one, quantify_wr_rr_reason

# get dir of repo (parent dir)
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Learner:
    """Implements a ML learner (based on PyTorch model)."""

    def __init__(
        self,
        model,
        loss,
        optimizer,
        device,
        modelname,
        load=False,
        base_criterion=F.cross_entropy,
    ):
        """
        Args:
            model: pytorch model.
            loss: pytorch loss function.
            optimizer: pytorch optimizer.
            device: either 'cuda' or 'cpu'.
            modelname: a unique name that identifies the model. Is used to store log/run files
                as well as the model itself.
            load: if True loads the model with modelname from model_store.
            base_criterion: pytorch functional loss function. Is used in experiments which
                disbale the XIL loss. Default cross_entropy.
        """
        self.model = model
        self.loss = loss
        self.base_criterion = base_criterion
        self.optimizer = optimizer
        self.device = device
        self.modelname = modelname
        self.run_id = str(uuid.uuid1())
        self.log_folder = f"{repo_dir}/training_and_eval/logs/" + self.modelname
        self.writer = SummaryWriter(
            log_dir=f"{repo_dir}/training_and_eval/runs/"
            + self.modelname
            + "--"
            + self.run_id,
            comment="_" + self.modelname + "_id_{}".format(self.run_id),
        )
        self.output_dir = f"{repo_dir}/training_and_eval/eval_output/{modelname}"
        os.makedirs(self.output_dir, exist_ok=True)
        if load:
            self.load(modelname + ".pt")

    def score(self, dataloader, criterion, verbose=False):
        """Returns the acc and loss on the specified dataloader."""
        size = len(dataloader.dataset)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data in dataloader:
                X, y = data[0].to(self.device), data[1].to(self.device)
                output = self.model(X)
                # output = F.softmax(self.model(X), dim=1)
                test_loss += criterion(output, y).item()
                correct += (output.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        if verbose:
            print(f"Test Error: Acc: {100*correct:>0.1f}%, Avg loss: {test_loss:>8f}")
        return 100 * correct, test_loss

    def save_learner(self, epochs=0, verbose=1, best=False):
        """Save the model dict to disk."""
        # if best:
        #    self.modelname = self.modelname + "-bestOnTrain"
        results = {
            "weights": self.model.state_dict(),
            "optimizer_dict": self.optimizer.state_dict(),
            "modelname": self.modelname,
            "loss": str(self.loss),
            "epochs": epochs,
            "rng_state": torch.get_rng_state(),
        }
        torch.save(
            results,
            f"{repo_dir}/training_and_eval/learner/model_store/"
            + self.modelname
            + ".pt",
        )
        if verbose == 1:
            print("Model saved!")

    def load(self, name):
        """Load the model with name from the model_store."""
        checkpoint = torch.load(
            f"{repo_dir}/training_and_eval/learner/model_store/" + name,
            map_location=torch.device(self.device),
        )
        self.model.load_state_dict(checkpoint["weights"])
        epochs_ = "none"
        if "optimizer_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"].type(torch.ByteTensor))
        if "epochs" in checkpoint:
            epochs_ = checkpoint["epochs"]
        print(
            f"Model {self.modelname} loaded! Was trained on {checkpoint['loss']} for {epochs_} epochs!"
        )

    def fit(
        self,
        dataloader,
        val_dataloader,
        epochs,
        save_best=False,
        save_last=True,
        verbose=True,
        verbose_after_n_epochs=1,
    ):
        """
        Fits the learner using training data from dataloader for specified number
        of epochs. After every epoch the learner is evaluated on the specified
        test_dataloader. Uses pytorch SummaryWriter to log stats for tensorboard.
        Writes the training progress and stats per epoch in a logfile to the logs-folder.

        Args:
            dataloader: train dataloader (X, y, expl) where expl are the ground-truth user
                feedback masks (optional).
            val_dataloader: validation dataloader (Xt, yt).
            epochs: number of epochs to train.
            save_best: saves the best model on the train loss to file.
            save_last: saves the model after every epoch .
            verbose: set to False if you want no print outputs.
            verbose_after_n_epochs: print outputs after every n epochs.
        """
        print("Start training...")
        with open(f"{self.log_folder}.log", "w") as f:
            f.write("epoch,acc,loss,ra_loss,rr_loss,val_acc,val_loss,time\n")
            best_val_loss = 100000
            elapsed_time = 0

            for epoch in range(1, epochs + 1):
                # how_many_rawr_epoch = torch.tensor([0])

                self.model.train()
                size = len(dataloader.dataset)
                train_loss, correct = 0, 0
                start_time = time.time()

                for batch, data in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    # functionality to disable xil_loss for n number
                    # of epochs beginning from first epoch
                    X, y = data[0].to(self.device), data[1].to(self.device)
                    output = self.model(X)
                    loss = self.base_criterion(output, y)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()

                    # for calculating loss, acc per epoch
                    train_loss += loss.item()
                    correct += (output.argmax(1) == y).type(torch.float).sum().item()

                train_loss /= size
                correct /= size

                train_acc = 100.0 * correct

                end_time = time.time()
                elapsed_time_cur = end_time - start_time
                elapsed_time += elapsed_time_cur

                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Acc/train", train_acc, epoch)
                self.writer.add_scalar("Time/train", elapsed_time_cur, epoch)

                val_acc, val_loss = self.score(val_dataloader, self.base_criterion)

                # printing acc and loss
                if verbose and (epoch % verbose_after_n_epochs == 0):
                    print(f"Epoch {epoch}| ", end="")
                    print(
                        f"Tests: accuracy: {(train_acc):>0.1f}%, loss: {train_loss:>8f} | ",
                        end="",
                    )
                    print(f"Val: accuracy: {val_acc:>0.1f}%, loss: {val_loss:>8f}")
                    # print(f" --number RAWR [{how_many_rawr_epoch}]")

                # test acc on test set
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Acc/val", val_acc, epoch)
                # write in logfile -> we need the logfile to see plots in Jupyter notebooks
                f.write(
                    f"{epoch},{(train_acc):>0.1f},{train_loss:>8f},{(val_acc):>0.1f},{val_loss:>8f},{elapsed_time_cur:>0.4f}\n"
                )
                # save the current best model on val set
                if save_best and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_learner(verbose=False, best=True, epochs=epoch)

                if save_last:
                    self.save_learner(verbose=False, epochs=epoch)

        print(f"--> Training took {elapsed_time:>4f} seconds!")
        self.writer.flush()
        self.writer.close()

        # return the train_acc and loss after last epoch
        return train_acc, train_loss, elapsed_time

    def validation_statistics(
        self,
        dataloader,
        criterion=F.cross_entropy,
        class_labels=None,
        save=True,
        worst_group=False,
    ):
        """Calculates confusion matrix and different statistics on dataloader."""

        size = len(dataloader.dataset)
        y_pred_list = []
        class_zero_pred_prob_list = []
        y_val_list = []
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data in dataloader:
                X, y = data[0].to(self.device), data[1].to(self.device)
                output = self.model(X)
                sigmoid_output = torch.sigmoid(output)
                pred_prob_class_zero = sigmoid_output[:, 0]
                _, preds = output.max(1)
                test_loss += criterion(output, y).item()
                correct += (output.argmax(1) == y).type(torch.float).sum().item()
                y_val_list += y.detach().cpu().tolist()
                y_pred_list += preds.detach().cpu().tolist()
                class_zero_pred_prob_list += (
                    pred_prob_class_zero.detach().cpu().tolist()
                )
        test_loss /= size
        correct /= size

        cm = confusion_matrix(y_val_list, y_pred_list, labels=class_labels)
        cr = classification_report(y_val_list, y_pred_list, labels=class_labels)
        print(f"Acc: {100*correct:>0.1f}%, Avg loss: {test_loss:>8f}")
        print("\nCONFUSION MATRIX ------------\n")
        print(f"{cm}")
        print("\nCLASSIFICATION REPORT ------------\n")
        print(f"{cr}")

        if save:
            if worst_group:
                wg_flag = "worst-group-"
            else:
                wg_flag = ""
            with open(
                f"{self.output_dir}/{self.modelname}-STATS-test-{wg_flag}accuracy.txt",
                "w",
            ) as f:
                f.write(f"Acc: {100*correct:>0.1f}%, Avg loss: {test_loss:>8f}\n")
                f.write("Confusion matrix:\n")
                f.write(f"{cm}\n")
                f.write("Classification report:\n")
                f.write(f"{cr}")

    def get_wr_metric(self, dataloader):
        avg = []
        explain_with_ig_one_by_one(
            self.model,
            dataloader,
            save_name=f"{self.output_dir}/IG-activations",
            next_to_each_other=True,  # was False
            device=self.device,
        )

        thresh = quantify_wr_rr_reason(
            dataloader,
            self.model,
            self.device,
            foldername=f"{self.output_dir}/wr",
            metric="wr",
            threshold=None,
            mode="mean",
            flags=True,
        )
        avg.append(
            quantify_wr_rr_reason(
                dataloader,
                self.model,
                self.device,
                foldername=f"{self.output_dir}/wr",
                metric="wr",
                threshold=thresh,
                mode="mean",
                flags=True,
            )
        )

        f = open(f"{self.output_dir}/{self.modelname}-STATS-test-wr.txt", "w")
        f.write(f"IG:\nmean:{np.mean(avg)}\n std:{np.std(avg)}")
        f.close()

    def get_rr_metric(self, dataloader):
        avg = []

        thresh = quantify_wr_rr_reason(
            dataloader,
            self.model,
            self.device,
            foldername=f"{self.output_dir}/rr",
            metric="rr",
            threshold=None,
            mode="mean",
            flags=True,
        )
        avg.append(
            quantify_wr_rr_reason(
                dataloader,
                self.model,
                self.device,
                foldername=f"{self.output_dir}/rr",
                metric="rr",
                threshold=thresh,
                mode="mean",
                flags=True,
            )
        )

        f = open(f"{self.output_dir}/{self.modelname}-STATS-test-rr.txt", "w")
        f.write(f"IG:\nmean:{np.mean(avg)}\n std:{np.std(avg)}")
        f.close()
