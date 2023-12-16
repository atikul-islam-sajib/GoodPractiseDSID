import sys
import argparse
import logging
import torch

sys.path.append("./alzheimer")

from data.data_loader import Dataloader
from features.build_features import FeatureBuilder
from models.model import Classifier
from models.train_model import Trainer
from visualization.visualize import ChartManager

logging.basicConfig(
    level=logging.INFO,
    filename="../GoodPractiseDSID/logs/clf.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    """
    Main function to train a classifier model for disease classification.

    This script takes arguments for dataset path, batch size, model specification,
    number of epochs, learning rate, and device selection. It uses these parameters
    to train a classifier using the specified dataset.
    """
    parser = argparse.ArgumentParser(description="Disease Classifier Training Script")

    parser.add_argument("--dataset", type=str, help="Provide the dataset path")
    parser.add_argument(
        "--augmentation",
        type=int,
        help="Provide the number of augmented build features",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Provide the batch size"
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help="Specify to use the model for classification",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "mps"],
        help="Select device to use: 'cpu', 'gpu', or 'mps'",
    )
    parser.add_argument(
        "--get_all_metrics",
        action="store_true",
        help="Model performance".capitalize(),
    )
    parser.add_argument(
        "--get_all_charts",
        action="store_true",
        help="Model charts and performance".capitalize(),
    )

    args = parser.parse_args()

    try:
        if args.device == "gpu" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif args.device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        logging.info(f"Using device: {device}")

        if args.model:
            try:
                logging.info("Creating the data loader".title())
                loader = Dataloader(zip_file=args.dataset)
                loader.unzip_dataset()

                logging.info("Creating the augmentation dataset".title())
                if args.augmentation:
                    build_features = FeatureBuilder(
                        augmentation=True, samples=args.augmentation
                    )

                    logging.info("Completing the augmentation dataset".title())
                else:
                    build_features = FeatureBuilder()

                build_features.build_feature()
                loader.extract_feature()
            except Exception as e:
                logging.error(f"Error during data loading and feature extraction: {e}")

            try:
                logging.info("Creating the classifier".title())
                clf = Classifier().to(device)
            except Exception as e:
                logging.error(f"Error during classifier creation: {e}")

            try:
                logging.info("Training the classifier".title())
                trainer = Trainer(classifier=clf, device=device, lr=args.lr)
                trainer.train(epochs=args.epochs)

            except Exception as e:
                logging.error(f"Error during evaluation: {e}")

        if args.get_all_metrics and args.device:
            try:
                clf = Classifier().to(device)
                trainer = Trainer(classifier=clf, device=device, lr=args.lr)
                model_evaluation, model_clf_report = trainer.model_performance()

                print(model_evaluation, "\n\n")
                print(model_clf_report)
            except Exception as e:
                print(e)
                logging.error(f"Error during evaluation: {e}")

        if args.get_all_charts:
            logging.info("Generating the charts".capitalize())
            try:
                visualizer = ChartManager()
            except Exception:
                logging.exception("Error during chart creation".capitalize())
            else:
                visualizer.plot_image_predictions()
                visualizer.plot_training_history()
                visualizer.plot_confusion_metrics()

            logging.info("Charts generated successfully".capitalize())
        else:
            logging.exception("Cannot generate the charts".capitalize())

    except Exception as e:
        print(e)
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
