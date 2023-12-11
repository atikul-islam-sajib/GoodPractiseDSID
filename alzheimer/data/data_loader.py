import argparse
import logging
import zipfile
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("./alzheimer")

from features.build_features import FeatureBuilder

logging.basicConfig(
    level=logging.INFO,
    filename="../GoodPractiseDSID/logs/dataloader.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Dataloader:
    """
    DataLoader class for unzipping a dataset and extracting features.

    Args:
        zip_file (str): Path to the zip file containing the dataset.

    Attributes:
        zip_file (str): The path to the zip file containing the dataset.
        X (list): List to store independent variables.
        y (list): List to store dependent variables.
        input_channel (int): Number of input channels.
        batch_size (int): Batch size for DataLoader.

    Methods:
        unzip_dataset: Unzips the dataset to the specified directory.
        extract_feature: Loads and extracts features from the dataset.
        split_dataset: Splits the dataset into training and testing sets.
        create_data_loader: Creates data loaders for training and testing data.
    """

    def __init__(self, zip_file):
        """
        Initializes a DataLoader instance.

        Args:
            zip_file (str): Path to the zip file containing the dataset.
        """
        self.zip_file = zip_file
        self.X = list()
        self.y = list()
        self.batch_size = 64

    def unzip_dataset(self):
        """
        Unzips the dataset to the specified directory and logs the process.
        """
        logging.info("Unzip is in progress")
        with zipfile.ZipFile(file=self.zip_file, mode="r") as zip_ref:
            zip_ref.extractall("../GoodPractiseDSID/data/raw/")

        logging.info("Unzip completed successfully")

    def extract_feature(self):
        """
        Extracts features from the dataset and prepares data loaders.
        """
        try:
            dataset = torch.load("../GoodPractiseDSID/data/raw/data.pth")
        except FileNotFoundError:
            logging.exception("Pickle File not found")
        else:
            for independent, dependent in dataset:
                self.X.append(independent)
                self.y.append(dependent)

            X_train, X_test, y_train, y_test = self.split_dataset(X=self.X, y=self.y)
            train_loader, test_loader = self.create_data_loader(
                X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
            )
            try:
                self._store_data_loader(
                    train_loader=train_loader, test_loader=test_loader
                )
            except FileNotFoundError:
                logging.exception("Data Folder not found")

    def split_dataset(self, **dataset):
        """
        Splits the dataset into training and testing sets.

        Args:
            dataset (dict): Dictionary containing X (independent variables) and y (dependent variables).

        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        X = np.array(dataset["X"]) / 255
        y = np.array(dataset["y"])

        return train_test_split(X, y, test_size=0.25, random_state=42)

    def create_data_loader(self, **dataset):
        """
        Creates data loaders for training and testing data.

        Args:
            dataset (dict): Dictionary containing X_train, X_test, y_train, y_test.

        Returns:
            Tuple: train_loader, test_loader
        """
        X_train = dataset["X_train"].reshape(
            dataset["X_train"].shape[0],
            dataset["X_train"].shape[3],
            dataset["X_train"].shape[1],
            dataset["X_train"].shape[2],
        )
        X_test = dataset["X_test"].reshape(
            dataset["X_test"].shape[0],
            dataset["X_test"].shape[3],
            dataset["X_test"].shape[1],
            dataset["X_test"].shape[2],
        )
        y_train = dataset["y_train"]
        y_test = dataset["y_test"]

        X_train = torch.tensor(data=X_train, dtype=torch.float32)
        X_test = torch.tensor(data=X_test, dtype=torch.float32)
        y_train = torch.tensor(data=y_train, dtype=torch.long)
        y_test = torch.tensor(data=y_test, dtype=torch.long)

        train_loader = DataLoader(
            dataset=list(zip(X_train, y_train)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            dataset=list(zip(X_test, y_test)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        return train_loader, test_loader

    def _store_data_loader(self, **dataset):
        train_loader = dataset["train_loader"]
        test_loader = dataset["test_loader"]

        # Define the directory path
        directory = "../GoodPractiseDSID/data/processed/"

        # Check if the directory exists, and create it if not
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the train_loader and test_loader as separate files
        [
            torch.save(dataset, os.path.join(directory, "train_loader.pth"))
            if index == 0
            else torch.save(dataset, os.path.join(directory, "test_loader.pth"))
            for index, dataset in enumerate([train_loader, test_loader])
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Loader for unzipping the dataset"
    )
    parser.add_argument("--dataset", type=str, help="Provide the dataset path")

    args = parser.parse_args()

    if args.dataset:
        logging.info("Data Loader is on processing.".title())
        loader = Dataloader(zip_file=args.dataset)
        loader.unzip_dataset()

        logging.info(
            "Data Loader is done with unzip & on the process of extracting features".title()
        )
        build_features = FeatureBuilder()
        build_features.build_feature()
        loader.extract_feature()

        logging.info("Data Loader is done with extracting features".title())

    else:
        logging.info("Please provide the dataset path".title())
