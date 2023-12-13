import logging
import sys
import argparse
import torch
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.append("./alzheimer")

logging.basicConfig(
    level=logging.INFO,
    filename="../GoodPractiseDSID/logs/evaluation.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Prediction:
    """
    A class for making predictions and computing various metrics on a given dataset.

    This class is designed to handle prediction tasks, specifically for datasets related to
    Alzheimer's disease. It includes methods for calculating accuracy, precision, recall, F1 score,
    generating a classification report, and computing a confusion matrix.

    Attributes:
        test_loader (str): Path to the test dataset loader file.
        train_loader (str): Path to the train dataset loader file.

    Methods:
        compute_accuracy(pred, target):
            Computes the accuracy of the predictions.

        compute_precision(pred, target):
            Computes the precision score of the predictions.

        compute_recall(target, pred):
            Computes the recall score of the predictions.

        compute_f1(target, pred):
            Computes the F1 score of the predictions.

        compute_classification_report(target, pred):
            Generates a classification report for the predictions.

        compute_confusion_matrix(target, pred):
            Computes a confusion matrix for the predictions.

        predict():
            Method to be implemented for making predictions.
    """

    def __init__(
        self, test_loader=None, train_loader=None, best_model=None, device=None
    ):
        """
        Initializes the Prediction class with test and train data loaders.

        Parameters:
            test_loader (str): Path to the test dataset loader file.
            train_loader (str): Path to the train dataset loader file.
        """
        self.test_loader = "../GoodPractiseDSID/data/processed/test_loader.pth"
        self.train_loader = "../GoodPractiseDSID/data/processed/train_loader.pth"
        self.device = device
        if best_model is None:
            try:
                self.model = "./alzheimer/checkpoint/model_{}.pth".format(
                    len(os.listdir("./alzheimer/checkpoint/")) - 1
                )
                self.model = torch.load(self.model)
            except Exception as e:
                logging.exception("model is not found".capitalize())
        else:
            self.model = best_model

    def load_data(self, data):
        return torch.load(data)

    def compute_accuracy(self, pred, target):
        """
        Computes the accuracy of predictions.

        Parameters:
            pred (list or ndarray): Predicted labels.
            target (list or ndarray): True labels.

        Returns:
            float: The accuracy score.
        """
        return accuracy_score(target, pred)

    def compute_precision(self, pred, target):
        """
        Computes the precision score of the predictions.

        Parameters:
            pred (list or ndarray): Predicted labels.
            target (list or ndarray): True labels.

        Returns:
            float: The precision score.
        """
        return precision_score(target, pred, average="micro")

    def compute_recall(self, target, pred):
        """
        Computes the recall score of the predictions.

        Parameters:
            target (list or ndarray): True labels.
            pred (list or ndarray): Predicted labels.

        Returns:
            float: The recall score.
        """
        return recall_score(target, pred, average="micro")

    def compute_f1(self, target, pred):
        """
        Computes the F1 score of the predictions.

        Parameters:
            target (list or ndarray): True labels.
            pred (list or ndarray): Predicted labels.

        Returns:
            float: The F1 score.
        """
        return f1_score(target, pred, average="micro")

    def compute_classification_report(self, target, pred):
        """
        Generates a classification report for the predictions.

        Parameters:
            target (list or ndarray): True labels.
            pred (list or ndarray): Predicted labels.

        Returns:
            str: A text report showing the main classification metrics.
        """
        return classification_report(target, pred)

    def compute_confusion_matrix(self, target, pred):
        """
        Computes a confusion matrix for the predictions.

        Parameters:
            target (list or ndarray): True labels.
            pred (list or ndarray): Predicted labels.

        Returns:
            ndarray: A confusion matrix of shape (n_classes, n_classes).
        """
        return confusion_matrix(target, pred)

    def save_results(self, **data):
        """
        Saves the actual and predicted labels to disk.

        Args:
            data (dict): A dictionary containing the 'actual' and 'predict' label data.
                        'actual' refers to the ground truth labels, and 'predict' refers
                        to the model's predicted labels.
        """
        actual, predict = data["actual"], data["predict"]
        [
            torch.save(
                data,
                "../GoodPractiseDSID/alzheimer/output/{}.pth".format(name),
            )
            for name, data in [("actual_label", actual), ("predict_label", predict)]
        ]

    def model_evaluation(self):
        """
        Executes the prediction process on the test dataset using the pre-loaded model.

        This method iterates through the test dataset loaded from the specified path in the class constructor.
        Each data point from the test set is fed into the model to generate predictions. The method ensures
        that the data and model are on the appropriate computing device (using Apple's Metal Performance Shaders (MPS)
        for acceleration if available, otherwise defaulting to CPU). It handles the nuances of device compatibility,
        tensor manipulation, and extraction of the predicted labels.

        The method is designed to accumulate predictions and their corresponding true labels from the entire test dataset,
        which can then be used for further evaluation and metrics calculation.

        Precondition:
            - The model must be loaded and compatible with the test data format and dimensions.
            - The test_loader attribute should point to a valid data loader with test data.

        Post condition:
            - The method returns a tuple containing arrays of predicted labels and actual labels.

        Returns:
            tuple of (predictions, actual_labels):
                predictions (list of ndarray): The array of predicted labels for the entire test dataset.
                actual_labels (list of ndarray): The array of actual labels corresponding to the test dataset.

        Raises:
            RuntimeError: If there is an issue in model prediction, possibly due to incompatible data format or model issues.
        """
        predictions = []
        actual_labels = []
        test_loader = torch.load(self.test_loader)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        for data, label in test_loader:
            logging.info("Model is predicting.".capitalize())

            predicted = self.model(data.to(device))
            predicted = torch.argmax(predicted, dim=1)

            predicted = predicted.detach().cpu().numpy()
            actual_label = label.detach().cpu().numpy()

            logging.info(
                "Model has finished predicting & store the labels and predicted".capitalize()
            )
            predictions.extend(predicted)
            actual_labels.extend(actual_label)

        logging.info("Store the labels and predicted".capitalize())
        try:
            self.save_results(predict=predictions, actual=actual_labels)
        except Exception as e:
            print(e)
            logging.exception("Result data cannot be saved.".capitalize())

        return predictions, actual_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict the labels for the test data."
    )
    parser.add_argument(
        "--predict", action="store_true", help="predict the model".capitalize()
    )

    args = parser.parse_args()

    if args.predict:
        prediction = Prediction()
        predictions, actual_labels = prediction.model_evaluation()

        logging.info(
            "ACCURACY # {} ".format(
                prediction.compute_accuracy(pred=predictions, target=actual_labels)
            )
        )
        logging.info(
            "PRECISION # {} ".format(
                prediction.compute_precision(pred=predictions, target=actual_labels)
            )
        )
        logging.info(
            "RECALL # {} ".format(
                prediction.compute_recall(pred=predictions, target=actual_labels)
            )
        )
        logging.info(
            "F1_SCORE # {} ".format(
                prediction.compute_f1(pred=predictions, target=actual_labels)
            )
        )
