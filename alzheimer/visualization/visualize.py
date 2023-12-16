import sys
import argparse
import logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append("./alzheimer")
import config_file

logging.basicConfig(
    level=logging.INFO,
    filename=config_file.CHARTS_LOG,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class ChartManager:
    """
    Manages the creation and display of various charts for model evaluation.

    This class handles the loading of model output data, including actual labels,
    predicted labels, dataset images, and training history. It provides methods to
    plot image predictions, training history, and confusion matrices.

    Attributes:
        actual (Tensor): Actual labels loaded from a .pth file.
        predicted (Tensor): Predicted labels loaded from a .pth file.
        dataset (Tensor): Image dataset loaded from a .pth file.
        history (dict): Training history data loaded from a .pth file.
    """

    def __init__(self):
        """
        Initializes the ChartManager by loading necessary data from .pth files.
        The data includes actual labels, predicted labels, dataset images, and training history.
        """
        logging.info("Initialise the data".capitalize())
        self.actual = torch.load(config_file.ACTUAL)
        self.predicted = torch.load(config_file.PREDICTED)
        self.dataset = torch.load(config_file.DATASET)
        self.history = torch.load(config_file.HISTORY)

    def plot_image_predictions(self):
        """
        Plots a grid of images from the dataset with their actual and predicted labels.

        The method reshapes the dataset, matches images with their actual and predicted labels,
        and plots them in a specified grid layout. The images are displayed in grayscale.
        """
        dataset = self.dataset.reshape(
            self.dataset.shape[0],
            self.dataset.shape[2],
            self.dataset.shape[3],
            self.dataset.shape[1],
        )
        actual_label = self.actual[: dataset.shape[0]]
        predicted_label = self.predicted[: dataset.shape[0]]
        number_of_rows = 5
        number_of_columns = 8

        plt.figure(figsize=(24, 12))

        for index, image in enumerate(dataset):
            plt.subplot(number_of_rows, number_of_columns, index + 1)
            plt.title(
                "Actual - {} \n Predicted - {}".format(
                    "AD".lower()
                    if actual_label[index] == 0
                    else "CONTROL".lower()
                    if actual_label[index] == 1
                    else "PD".lower(),
                    "AD".title()
                    if predicted_label[index] == 0
                    else "CONTROL".title()
                    if predicted_label[index] == 1
                    else "PD".title(),
                )
            )
            plt.imshow(image, cmap="gray")
            plt.axis("off")

        plt.savefig("../GoodPractiseDSID/alzheimer/figures/image_prediction.png")
        plt.show()

    def plot_training_history(self):
        """
        Plots the training and validation loss and accuracy from the training history.

        This method creates a subplot with two graphs: one for training vs. validation loss
        and another for training vs. validation accuracy. It helps in visualizing the
        performance of the model over the course of training epochs.
        """
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        axes[0].plot(self.history["m1_train_loss"], label="train_loss")
        axes[0].plot(self.history["m1_val_loss"], label="val_loss")
        axes[0].set_title("train vs val loss".capitalize())
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(self.history["m1_train_acc"], label="train_acc")
        axes[1].plot(self.history["m1_val_acc"], label="val_acc")
        axes[1].set_title("train vs val accuracy".capitalize())
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        plt.savefig("../GoodPractiseDSID/alzheimer/figures/training_history.png")
        plt.show()

    def plot_confusion_metrics(self):
        """
        Plots a confusion matrix using the actual and predicted labels.

        This method generates a confusion matrix from the actual and predicted labels
        and visualizes it as a heatmap, providing insights into the model's performance.
        """
        cm = confusion_matrix(self.actual, self.predicted)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="g", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title("Confusion Matrix")

        plt.savefig("../GoodPractiseDSID/alzheimer/figures/confusion_metrics.png")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating the charts".capitalize())

    parser.add_argument(
        "--get_all_charts",
        action="store_true",
        help="Model charts and performance".capitalize(),
        required=True,
    )

    args = parser.parse_args()

    if args.get_all_charts:
        logging.info("Generating the charts".capitalize())

        visualizer = ChartManager()
        visualizer.plot_image_predictions()
        visualizer.plot_training_history()
        visualizer.plot_confusion_metrics()

        logging.info("Charts generated successfully".capitalize())
    else:
        logging.exception("Cannot generate the charts".capitalize())
