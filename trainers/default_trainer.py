# Import necessary libraries for data manipulation, machine learning, and file handling
import copy
import os
import pickle
import numpy as np
import torch
from accelerate import (
    Accelerator,
)  # Simplifies running PyTorch models on multi-GPUs/TPUs
from sklearn.model_selection import KFold  # For k-fold cross-validation
from torch.optim import Adam  # Import the Adam optimizer
from torch.utils.data import (
    DataLoader,
    SubsetRandomSampler,
)  # For efficient data loading and sampling
from torch.utils.tensorboard import SummaryWriter  # For logging to TensorBoard
from tqdm import tqdm  # For displaying progress bars
from utils.helper import (
    get_accuracy,
    get_metrics,
)  # Import helper functions for metrics calculation


class Trainer:
    def __init__(self, cfg):
        # Initialize the Trainer with configuration settings, model preparation, and data loading strategies
        self.model = cfg.backbone  # Model configuration from cfg
        self.model.float().cuda()  # Ensure model uses float precision and moves to GPU if available
        self.batch_size = cfg.batch_size  # Batch size for training and validation
        self.early_stop_patience = (
            cfg.early_stop_patience
        )  # Early stopping patience for validation

        self.dataset = cfg.dataset  # Training dataset
        self.validate_every = cfg.validate_every  # Frequency of validation
        self.accelerator = Accelerator()  # Accelerator object for device-agnostic code

        # Setup for k-fold cross-validation if specified, else default to a single training/validation split
        if hasattr(cfg, "k"):
            self.k = cfg.k
            self.splits = KFold(n_splits=cfg.k, shuffle=True)
        else:
            self.k = 1
            self.splits = KFold(n_splits=1)

        print("Training + Validation set:", len(cfg.dataset))

        # Prepare test data loaders if test datasets are provided
        if hasattr(cfg, "dataset_test"):
            self.test_dls = []
            for dataset in cfg.dataset_test:
                test_dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
                test_dl = self.accelerator.prepare(
                    test_dl
                )  # Prepare DataLoader for acceleration
                self.test_dls.append(test_dl)
            print("Number of test sets:", len(self.test_dls))

        self.lr = cfg.lr  # Learning rate for optimizer

        # Setup for logging and saving model checkpoints
        self.log_writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, "training"))
        self.result_path = os.path.join(cfg.log_dir, "checkpoints")

        self.max_epoch = cfg.epochs  # Maximum number of training epochs

        # Define a function for converting labels to one-hot encoding
        self.one_hot_label = lambda label: torch.nn.functional.one_hot(
            label, num_classes=7
        ).float()

        self.loss_fn = torch.nn.CrossEntropyLoss()  # Loss function for training

        # Load model from checkpoint if specified
        if hasattr(cfg, "checkpoint"):
            self.load(cfg.checkpoint)

    def load(self, checkpoint):
        # Load model state from a checkpoint file
        state_dict = torch.load(checkpoint)
        self.model.load_state_dict(state_dict["model"])

    def save(self, fold):
        # Save model state, fold, and epoch information to a checkpoint file
        data = {"fold": fold, "epoch": self.epoch, "model": self.model.state_dict()}
        torch.save(data, os.path.join(self.result_path, f"Fold_{fold}.pt"))

    def inference(self):
        # Run inference on the test datasets and save the latent representations
        self.model.eval()  # Set the model to evaluation mode

        for idx, test_dl in enumerate(self.test_dls):
            print("Test set #" + str(idx))
            pred = [[] for i in range(7)]  # Initialize prediction list for each class

            with torch.no_grad():  # No gradient computation for inference
                for data in test_dl:
                    batch_label, batch_input = data
                    results = self.model.bypass_for_tsne(
                        torch.Tensor(batch_input).float()
                    )
                    results = (
                        results.cpu().tolist()
                    )  # Move results to CPU and convert to list
                    for i, label in enumerate(batch_label):
                        pred[label].append(results[i])  # Append results based on labels

            pred_path = os.path.join(self.result_path, f"Latent_{idx}.pkl")
            pickle.dump(
                pred, open(pred_path, "wb")
            )  # Save predictions to a pickle file

        self.model.train()  # Set the model back to training mode

    def cross_validate(self):
        # Perform k-fold cross-validation or a single training/validation run based on configuration
        if self.k == 1:
            # If k=1, perform a single training/validation run
            return self.train(
                DataLoader(self.dataset, self.batch_size, shuffle=True), None, fold=0
            )

        model_copy = copy.deepcopy(
            self.model
        )  # Make a deep copy of the model for cross-validation

        # Iterate over each fold in the k-fold split
        for fold, (train_idx, valid_idx) in enumerate(
            self.splits.split(np.arange(len(self.dataset)))
        ):
            print("Fold #" + str(fold))
            # Create DataLoaders for training and validation subsets
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            train_dl = DataLoader(self.dataset, self.batch_size, sampler=train_sampler)
            valid_dl = DataLoader(self.dataset, self.batch_size, sampler=valid_sampler)

            # Reset model to initial state and setup optimizer
            self.model.load_state_dict(model_copy.state_dict())
            self.opt = Adam(self.model.parameters(), lr=self.lr)

            # Prepare model, dataset, and DataLoaders for acceleration
            self.model, self.dataset, train_dl, valid_dl = self.accelerator.prepare(
                self.model, self.dataset, train_dl, valid_dl
            )

            self.train(train_dl, valid_dl, fold)  # Train the model on the current fold

        # Evaluate the model on test datasets after cross-validation
        for idx, test_dl in enumerate(self.test_dls):
            print("Test set #" + str(idx))
            for fold in range(self.k):
                saved_dict = torch.load(
                    os.path.join(self.result_path, f"Fold_{fold}.pt")
                )
                self.model.load_state_dict(saved_dict["model"])
                self.evaluate(test_dl)  # Evaluate the model on the test dataset
            print("")

    def validate(self, dl):
        # Validate the model on a given DataLoader and return the accuracy
        self.model.eval()  # Set the model to evaluation mode
        accuracy_sum = []  # Initialize list to accumulate accuracies

        with torch.no_grad():  # No gradient computation for validation
            for data in dl:
                batch_label, batch_input = data
                batch_label = self.one_hot_label(
                    batch_label
                )  # Convert labels to one-hot encoding
                results = self.model(torch.Tensor(batch_input).float())
                accuracy_sum.extend(
                    get_accuracy(results, batch_label)
                )  # Calculate accuracy

        acc = np.sum(accuracy_sum) / len(accuracy_sum)  # Compute average accuracy
        self.model.train()  # Set the model back to training mode
        return acc

    def evaluate(self, test_dl):
        # Evaluate the model on a test DataLoader and print metrics
        self.model.eval()  # Set the model to evaluation mode
        gt = []  # Ground truth labels
        pred = []  # Predicted labels

        with torch.no_grad():  # No gradient computation for evaluation
            for data in test_dl:
                batch_label, batch_input = data
                results = self.model(torch.Tensor(batch_input).float())
                gt.extend(batch_label.tolist())  # Append ground truth labels
                pred.extend(
                    torch.argmax(results, dim=-1).int().cpu().tolist()
                )  # Append predicted labels

        metrics = get_metrics(pred, gt)  # Calculate evaluation metrics
        for k in metrics.keys():
            print(k, metrics[k], end=" ")  # Print each metric
        print("")

        self.model.train()  # Set the model back to training mode

    def train(self, train_dl, valid_dl, fold):
        # Train the model with given DataLoaders for training and validation, and fold number
        self.epoch = 0  # Current epoch number
        max_val_acc = -1  # Maximum validation accuracy
        val_acc = 0  # Current validation accuracy
        early_stop_counter = 0  # Counter for early stopping

        with tqdm(
            initial=self.epoch, total=self.max_epoch
        ) as pbar:  # Progress bar for epochs
            for _ in range(self.max_epoch):
                loss_sum = []  # Initialize list to accumulate losses
                accuracy_sum = []  # Initialize list to accumulate accuracies

                for data in train_dl:
                    batch_label, batch_input = data
                    batch_label = self.one_hot_label(
                        batch_label
                    )  # Convert labels to one-hot encoding
                    results = self.model(torch.Tensor(batch_input).float())
                    accuracy_sum.extend(
                        get_accuracy(results, batch_label)
                    )  # Calculate accuracy

                    loss = self.loss_fn(results, batch_label)  # Calculate loss
                    self.accelerator.backward(loss)  # Compute gradients
                    loss_sum.append(loss.item())  # Append loss value

                    self.opt.step()  # Update model parameters
                    self.opt.zero_grad()  # Reset gradients

                loss_mean = np.mean(loss_sum)  # Compute mean loss
                self.log_writer.add_scalar(
                    "loss", loss_mean, global_step=self.epoch
                )  # Log loss

                accuracy = np.sum(accuracy_sum) / len(
                    accuracy_sum
                )  # Compute mean accuracy
                self.log_writer.add_scalar(
                    "accuracy", accuracy, global_step=self.epoch
                )  # Log accuracy

                pbar.set_description(
                    f"loss: {loss_mean:.6f} acc: {accuracy:.6f} val_acc: {val_acc:.6f}"
                )  # Update progress bar description
                pbar.update(1)  # Increment progress bar

                self.epoch += 1  # Increment epoch number

                if self.epoch % self.validate_every == 0 and valid_dl is not None:
                    # Perform validation every specified number of epochs
                    val_acc = self.validate(valid_dl)  # Validate model

                    if val_acc > max_val_acc:
                        # Update max validation accuracy and reset early stopping counter
                        max_val_acc = val_acc
                        early_stop_counter = 0
                        self.save(fold)  # Save model checkpoint
                    else:
                        # Increment early stopping counter
                        early_stop_counter += 1
                        if early_stop_counter > self.early_stop_patience:
                            # Stop training if early stop condition is met
                            break


def get_trainer(cfg):
    # Function to initialize and return a Trainer instance
    return Trainer(cfg)
