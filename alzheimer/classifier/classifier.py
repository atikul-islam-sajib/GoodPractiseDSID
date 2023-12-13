import sys
import argparse
import logging
import torch

sys.path.append("./alzheimer")

from data.data_loader import Dataloader
from features.build_features import FeatureBuilder
from models.model import Classifier
from models.train_model import Trainer


def main():
    """
    Main function to train a classifier model for disease classification.

    This script takes arguments for dataset path, batch size, model specification,
    number of epochs, learning rate, and device selection. It uses these parameters
    to train a classifier using the specified dataset.
    """
    parser = argparse.ArgumentParser(description="Disease Classifier Training Script")

    parser.add_argument(
        "--dataset", type=str, required=True, help="Provide the dataset path"
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

    args = parser.parse_args()

    try:
        # Set the device based on user input and device availability
        if args.device == "gpu" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif args.device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        logging.info(f"Using device: {device}")

        if args.model:
            loader = Dataloader(zip_file=args.dataset)
            build_features = FeatureBuilder()
            build_features.build_feature()
            loader.extract_feature()

            clf = Classifier().to(device)

            trainer = Trainer(classifier=clf, device=device, lr=args.lr)
            trainer.train(epochs=args.epochs)
            model_evaluation, model_clf_report = trainer.model_performance()

            print(model_evaluation)
            print(model_clf_report)

            logging.info("Model training and evaluation completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
