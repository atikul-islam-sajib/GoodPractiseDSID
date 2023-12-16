import logging
import argparse
import cv2
import os
import sys
import torch
import random

sys.path.append("./alzheimer")
import config_file
from augmentator.augmentation import Augmentation

logging.basicConfig(
    level=logging.INFO,
    filename="../GoodPractiseDSID/logs/features.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class FeatureBuilder:
    def __init__(self, augmentation=False, samples=1000):
        """
        Initialize the FeatureBuilder class.

        - Initialize a list to store image data.
        - Define image height and width for resizing.
        - Define categories for image classification.
        """
        self.augmentation = augmentation
        self.samples = samples
        self.store_image_data = list()
        self.image_height = 120
        self.image_width = 120
        self.categories = ["AD", "CONTROL", "PD"]

    def build_feature(self):
        """
        Build and process image features.

        - Iterate through train and test directories.
        - Iterate through categories within each directory.
        - Load and process images, resizing them to the specified dimensions.
        - Assign labels to images based on the category.
        - Store image data as [image, label] pairs in self.store_image_data.
        - Log progress and completion of folder processing.
        """
        if self.augmentation == True:
            try:
                for folder in ["train", "test"]:
                    augmentation = Augmentation(
                        samples=self.samples,
                        file_path=os.path.join(config_file.DATA_FOLDER_PATH, folder),
                    )
                    augmentation.build_augmentation()
            except Exception as e:
                logging.exception("Augmentation Error".capitalize())
            else:
                logging.info("Augmentation Completed".capitalize())

                train_directory = [
                    config_file.AUG_TRAIN,
                    config_file.AUG_TEST,
                ]
        else:
            train_directory = [
                config_file.TRAIN_DATA,
                config_file.TEST_DATA,
            ]

        logging.info("Building feature data...".capitalize())

        for directory in train_directory:
            for category in self.categories:
                folder_path = os.path.join(directory, category)

                for filename in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, filename)
                    image = cv2.imread(image_path)
                    if image is None:
                        logging.info(
                            "Image cannot be extracted due to corrupted image".title()
                        )
                    else:
                        image = cv2.resize(image, (self.image_height, self.image_width))
                        label = self.categories.index(category)
                        self.store_image_data.append([image, label])

                logging.info("{} - folder is completed".title().format(category))

            logging.info("{} - folder is completed".title().format(directory))

        try:
            random.shuffle(self.store_image_data)
            torch.save(
                self.store_image_data,
                config_file.DATA_PATH,
            )
        except FileNotFoundError as e:
            logging.info("File not found: {}".format(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing")

    parser.add_argument("--build_feature", help="Enter the path")

    args = parser.parse_args()

    if args.build_feature:
        feature_builder = FeatureBuilder()
        feature_builder.build_feature()
    else:
        # Log an exception if feature building is not successful.
        logging.exception(
            "Cannot be extracted features from images due to internal issue".title()
        )
