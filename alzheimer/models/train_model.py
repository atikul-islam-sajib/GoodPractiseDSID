import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

import sys

sys.path.append("./alzheimer")

from models.model import Classifier
from models.predict_model import Prediction

logging.basicConfig(
    level=logging.INFO,
    filename="../GoodPractiseDSID/logs/train.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Trainer:
    """
    Trainer class for training a machine learning classifier.

    Attributes:
        classifier (torch.nn.Module): Neural network model for classification.
        device (torch.device): Device on which the model will be trained (e.g., 'cuda' or 'cpu').
        model1_loss_function (torch.nn.Module): Loss function for the model.
        model1_lr (float): Learning rate for the optimizer.
        model1_optimizer (torch.optim.Optimizer): Optimizer used for training.
        history (dict): Dictionary to track training and validation metrics.
    """

    def __init__(self, classifier=None, device=None, lr=0.001):
        """
        Initializes the Trainer object.

        Args:
            classifier (torch.nn.Module): The classifier model to be trained.
            device (torch.device): The device to train the model on.

        Raises:
            ValueError: If classifier is not defined.
        """
        if classifier is None:
            raise ValueError("Classifier model is not defined")
        self.classifier = classifier.to(device)
        self.device = device

        self.model1_loss_function = nn.CrossEntropyLoss()
        self.model1_lr = lr
        self.model1_optimizer = optim.Adam(
            self.classifier.parameters(), lr=self.model1_lr
        )

        self.history = {
            "m1_train_loss": [],
            "m1_train_acc": [],
            "m1_val_loss": [],
            "m1_val_acc": [],
        }

    def _connect_GPU(self, independent_data, dependent_data):
        """
        Transfers data and labels to the specified device (e.g., GPU).

        Args:
            independent_data (torch.Tensor): Input features.
            dependent_data (torch.Tensor): Target labels.

        Returns:
            tuple: Tuple containing the input features and labels transferred to the device.
        """
        return independent_data.to(self.device), dependent_data.to(self.device)

    def _do_back_propagation(self, optimizer, model_loss, retain_graph=False):
        """
        Performs backpropagation and updates model parameters.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to use.
            model_loss (torch.Tensor): Computed loss of the model.
            retain_graph (bool): Whether to retain computation graph for further backward passes.
        """
        optimizer.zero_grad()
        model_loss.backward(retain_graph=retain_graph)
        optimizer.step()

    def _compute_predicted_label(self, model_prediction):
        """
        Computes predicted labels from model output.

        Args:
            model_prediction (torch.Tensor): The output from the model.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        return torch.argmax(model_prediction, dim=1).cpu().detach().numpy()

    def _compute_actual_label(self, actual_label):
        """
        Processes actual labels for comparison.

        Args:
            actual_label (torch.Tensor): Actual labels.

        Returns:
            numpy.ndarray: Processed actual labels.
        """
        return actual_label.cpu().detach().numpy()

    def _compute_model_loss(self, model, loss_function, actual_label):
        """
        Computes loss for the model.

        Args:
            model (torch.Tensor): The output from the model.
            loss_function (torch.nn.Module): Loss function to be used.
            actual_label (torch.Tensor): Actual labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        return loss_function(model, actual_label)

    def _model_accuracy(self, actual_label, predicted_label):
        """
        Computes accuracy of the model.

        Args:
            actual_label (numpy.ndarray): Actual labels.
            predicted_label (numpy.ndarray): Predicted labels by the model.

        Returns:
            float: Accuracy score.
        """
        return accuracy_score(actual_label, predicted_label)

    def _display(self, **data):
        """
        Displays training and validation metrics for the current epoch.

        Args:
            model1_train_loss (list): List of training losses for the epoch.
            model1_train_acc (float): Training accuracy for the epoch.
            model1_val_loss (list): List of validation losses for the epoch.
            model1_val_acc (float): Validation accuracy for the epoch.
            running_epochs (int): Current epoch.
            total_epochs (int): Total number of epochs.
        """
        print("Epochs: {}/{}".format(data["running_epochs"] + 1, data["total_epochs"]))
        print(
            "m1_loss: {:.4f} - m1_acc: {:.4f} - val1_loss: {:.4f} - val1_acc: {:.4f}".format(
                np.mean(data["model1_train_loss"]),
                data["model1_train_acc"],
                np.mean(data["model1_val_loss"]),
                data["model1_val_acc"],
            )
        )

    def save_models(self, model, epoch):
        try:
            torch.save(
                model,
                "./alzheimer/checkpoint/model_{}.pth".format(epoch),
            )
        except Exception as e:
            logging.exception("Saving model exception occurred".capitalize())

    def train(self, epochs):
        """
        Trains the model for a given number of epochs.

        Args:
            epochs (int): Number of epochs to train the model.
        """
        torch.autograd.set_detect_anomaly(True)

        train_loader = torch.load("../GoodPractiseDSID/data/processed/train_loader.pth")
        test_loader = torch.load("../GoodPractiseDSID/data/processed/test_loader.pth")

        #######################
        ##      Training     ##
        #######################

        logging.info("Model training measurement".capitalize())

        for epoch in range(epochs):
            model1_train_loss = []
            model1_train_pred = []
            model_actual_label = []

            for X_train_batch, y_train_batch in train_loader:
                X_train_batch, y_train_batch = self._connect_GPU(
                    X_train_batch, y_train_batch
                )

                model1 = self.classifier(X_train_batch)
                model1_loss = self._compute_model_loss(
                    model1, self.model1_loss_function, y_train_batch
                )
                self._do_back_propagation(
                    self.model1_optimizer, model1_loss, retain_graph=True
                )

                model1_predicted = self._compute_predicted_label(model1)

                logging.info("Store training accuracy & loss into history".capitalize())

                model_actual_label.extend(self._compute_actual_label(y_train_batch))
                model1_train_pred.extend(model1_predicted)
                model1_train_loss.append(model1_loss.item())

            model1_accuracy = self._model_accuracy(
                model_actual_label, model1_train_pred
            )

            self.history["m1_train_loss"].append(np.mean(model1_train_loss))
            self.history["m1_train_acc"].append(model1_accuracy)

            model1_val_loss = []
            model1_val_pred = []
            model_val_actual_label = []

            #######################
            ##     Validation    ##
            #######################

            logging.info("Validation is calculating".capitalize())

            self.classifier.eval()

            with torch.no_grad():
                for X_val_batch, y_val_batch in test_loader:
                    X_val_batch, y_val_batch = self._connect_GPU(
                        X_val_batch, y_val_batch
                    )

                    model1 = self.classifier(X_val_batch)
                    val_model1_loss = self._compute_model_loss(
                        model1, self.model1_loss_function, y_val_batch
                    )
                    model1_val_pred.extend(self._compute_predicted_label(model1))
                    model_val_actual_label.extend(
                        self._compute_actual_label(y_val_batch)
                    )
                    model1_val_loss.append(val_model1_loss.item())

            model1_val_accuracy = self._model_accuracy(
                model_val_actual_label, model1_val_pred
            )

            logging.info("Store validation accuracy & loss into history".capitalize())

            self.history["m1_val_loss"].append(np.mean(model1_val_loss))
            self.history["m1_val_acc"].append(model1_val_accuracy)

            logging.info("Validation is done".capitalize())

            self._display(
                model1_train_loss=model1_train_loss,
                model1_train_acc=model1_accuracy,
                model1_val_loss=model1_val_loss,
                model1_val_acc=model1_val_accuracy,
                running_epochs=epoch,
                total_epochs=epochs,
            )

            logging.info("Saving model1".capitalize())
            self.save_models(model=self.classifier, epoch=epoch)

    def model_performance(self):
        """
        Computes and returns key performance metrics of the model on the test dataset.

        This method evaluates the model's performance by calculating accuracy, precision, recall, and F1 score.
        It utilizes the Prediction class for generating predictions and computing metrics, and organizes the
        results into a Pandas DataFrame for easy analysis.

        Returns:
            pandas.DataFrame: A DataFrame with columns for Accuracy, Precision, Recall, and F1 score.
        """
        predictor = Prediction(device=self.device)
        predictions, actual_labels = predictor.model_evaluation()

        model_evaluation = pd.DataFrame(
            {
                "Accuracy".capitalize(): [
                    predictor.compute_accuracy(predictions, actual_labels)
                ],
                "Precision".capitalize(): [
                    predictor.compute_precision(predictions, actual_labels)
                ],
                "Recall".capitalize(): [
                    predictor.compute_recall(predictions, actual_labels)
                ],
                "F1_score".capitalize(): [
                    predictor.compute_f1(predictions, actual_labels)
                ],
            }
        )
        model_clf_report = predictor.compute_classification_report(
            predictions, actual_labels
        )
        return model_evaluation, model_clf_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate".title())

    args = parser.parse_args()

    if args.epochs and args.lr:
        logging.info("Start training".title())
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        logging.info("Using device: ".title())
        clf = Classifier().to(device)

        trainer = Trainer(classifier=clf, device=device, lr=args.lr)
        trainer.train(epochs=args.epochs)
        logging.info("Training is done".capitalize())

        logging.info("Start evaluation".title())
        model_evaluation, model_clf_report = trainer.model_performance()

        logging.info(model_evaluation)
        logging.info(model_clf_report)
    else:
        logging.error("Please provide number of epochs")
