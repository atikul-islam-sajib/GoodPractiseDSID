import argparse
import logging
import zipfile
import os
import sys
import joblib

sys.path.append("./alzheimer")

from features.build_features import FeatureBuilder

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join("../alzheimer/logs"),
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class DataLoader:
    """
    DataLoader class for unzipping a dataset.

    Args:
        zip_file (str): Path to the zip file containing the dataset.

    Methods:
        unzip_dataset: Unzips the dataset to the specified directory.
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

    def unzip_dataset(self):
        """
        Unzips the dataset to the specified directory and logs the process.
        """
        logging.info("Unzip is in progress")
        with zipfile.ZipFile(file=self.zip_file, mode="r") as zip_ref:
            zip_ref.extractall("../alzheimer/data/raw/")

        logging.info("Unzip completed successfully")

    def extract_feature(self):
        try:
            dataset = joblib.load(filename="../alzheimer/data/raw/data.pkl")
        except FileNotFoundError:
            logging.exception("Pickle File not found")
        else:
            for independent, dependent in dataset:
                self.X.append(independent)
                self.y.append(dependent)

            self.split_dataset(X=self.X, y=self.y)

    def split_dataset(self, **dataset):
        X = dataset["X"]
        y = dataset["y"]

        print(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Loader for unzipping the dataset"
    )
    parser.add_argument("--dataset", type=str, help="Provide the dataset path")

    args = parser.parse_args()

    if args.dataset:
        loader = DataLoader(zip_file=args.dataset)
        loader.unzip_dataset()
        loader.extract_feature()
