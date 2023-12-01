import argparse
import logging
import zipfile
import os

import config_file
from ..experiments.experiment import Experiments

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(config_file.LOGS_PATH, "dataloader.log"),
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

    def unzip_dataset(self):
        """
        Unzips the dataset to the specified directory and logs the process.
        """
        logging.info("Unzip is in progress")
        with zipfile.ZipFile(file=self.zip_file, mode="r") as zip_ref:
            zip_ref.extractall("../alzheimer/data/raw/")

        logging.info("Unzip completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Loader for unzipping the dataset"
    )
    parser.add_argument("--dataset", type=str, help="Provide the dataset path")

    args = parser.parse_args()

    if args.dataset:
        loader = DataLoader(zip_file=args.dataset)
        loader.unzip_dataset()
