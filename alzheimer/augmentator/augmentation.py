import sys
import os
import logging
import argparse
import Augmentor

sys.path.append("./alzheimer")

logging.basicConfig(
    level=logging.INFO,
    filename="../GoodPractiseDSID/logs/augmentation.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Augmentation:
    def __init__(self, samples=1000, file_path=None):
        self.p = Augmentor.Pipeline(file_path)
        self.samples = samples

    def build_augmentation(self):
        """
        Sets up an image augmentation pipeline using the Augmentor library.

        This method configures various augmentation techniques specifically tailored for
        Alzheimer's disease classification using brain scans. The augmentations include
        rotation, random cropping, resizing, random brightness adjustment, random contrast
        adjustment, and zooming. These augmentations are intended to introduce realistic
        variations to the dataset, aiding the CNN model in generalizing better to unseen data.
        """

        logging.info("Saving the augmentation dataset".capitalize())

        logging.info("Rotation".capitalize())
        self.p.rotate(probability=0.3, max_left_rotation=10, max_right_rotation=10)

        logging.info("Random Cropping".capitalize())
        self.p.crop_random(probability=0.1, percentage_area=0.5)

        logging.info("Resizing".capitalize())
        self.p.resize(probability=0.1, width=100, height=100)

        logging.info("Random Brightness".capitalize())
        self.p.random_brightness(probability=0.5, min_factor=0.4, max_factor=0.9)

        logging.info("Random Contrast".capitalize())
        self.p.random_contrast(probability=0.5, min_factor=0.9, max_factor=1.4)

        logging.info("Zoom".capitalize())
        self.p.zoom(probability=0.7, min_factor=1.1, max_factor=1.5)

        try:
            self.p.sample(self.samples)
        except Exception as e:
            logging.exception("Augmentation cannot be possible".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment the dataset")
    parser.add_argument("--augmentation", type=int, help="Path to the dataset")

    args = parser.parse_args()
    if args.augmentation:
        aug = Augmentation(samples=1)
        aug.build_augmentation()

    else:
        logging.exception("Exception in the building augmentation dataset".capitalize())
