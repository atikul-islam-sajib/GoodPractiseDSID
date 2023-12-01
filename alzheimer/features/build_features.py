import logging
import argparse
import cv2
import os
import joblib
import random

logging.basicConfig(
    level=logging.INFO,
    filename="feature.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class FeatureBuilder:
    def __init__(self):
        """
        Initialize the FeatureBuilder class.

        - Initialize a list to store image data.
        - Define image height and width for resizing.
        - Define categories for image classification.
        """
        self.store_image_data = list()
        self.image_height = 80
        self.image_width = 80
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
        - Shuffle the image data and save it to a pickle file.
        """
        train_directory = [
            "../alzheimer/data/raw/dataset/train",
            "../alzheimer/data/raw/dataset/test",
        ]
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
            # Shuffle the image data and save it to a pickle file.
            joblib.dump(
                value=random.shuffle(self.store_image_data),
                filename="../alzheimer/data/raw/data.pkl",
            )
        except PickleError as e:
            logging.info("PickleError: {}".format(e))


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
