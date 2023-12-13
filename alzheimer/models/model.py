import argparse
import logging
import sys
import torch
import torch.nn as nn
from collections import OrderedDict

sys.path.append("./alzheimer")


logging.basicConfig(
    level=logging.INFO,
    filename="../GoodPractiseDSID/logs/model.log",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Classifier(nn.Module):
    """
    The Classifier class is a PyTorch model used for classifying images.
    It consists of three parallel convolutional branches (left, middle, right)
    followed by fully connected layers.
    """

    def __init__(self):
        super().__init__()

        logging.info("Classifier initialized: Define the left convolutional branch")

        self.left_conv = self.make_conv_layers(
            layers=[
                (3, 32, 3, 1, 1, 2, 2, 0.0),
                (32, 16, 3, 1, 1, 2, 2, 0.5),
                (16, 8, 3, 1, 1, 2, 2, 0.5),
            ],
            prefix="left",
        )

        logging.info("Classifier initialized: Define the middle convolutional branch")

        self.middle_conv = self.make_conv_layers(
            layers=[
                (3, 32, 4, 1, 1, 2, 2, 0.0),
                (32, 16, 4, 1, 1, 2, 2, 0.4),
                (16, 8, 4, 1, 1, 2, 2, 0.2),
            ],
            prefix="middle",
        )

        logging.info("Classifier initialized: Define the right convolutional branch")

        self.right_conv = self.make_conv_layers(
            layers=[
                (3, 32, 5, 1, 1, 2, 2, 0.0),
                (32, 16, 5, 1, 1, 2, 2, 0.3),
                (16, 8, 5, 1, 1, 2, 2, 0.3),
            ],
            prefix="right",
        )

        logging.info(
            "Classifier initialized: Define the combined layer after concatenating the outputs from the three branches"
        )

        self.combined_layer = self.make_combined_layer(
            layers=[(15 * 15 * 8 + 14 * 14 * 8 + 13 * 13 * 8, 256)], prefix="combined"
        )

        logging.info(
            "Classifier initialized: Define fully connected layers for each branch after the combined layer"
        )

        self.left_fc = self.make_fc_layers(
            layers=[(256, 128, 0.3), (128, 64, 0.4), (64, 16, 0.3), (16, 3)],
            prefix="left_fc",
        )
        self.middle_fc = self.make_fc_layers(
            layers=[(256, 64, 0.4), (64, 32, 0.4), (32, 3)],
            prefix="middle_fc",
        )
        self.right_fc = self.make_fc_layers(
            layers=[(256, 32, 0.4), (32, 16, 0.3), (16, 3)],
            prefix="right_fc",
        )

    def make_conv_layers(self, layers, prefix):
        """
        Creates a series of convolutional layers.

        Args:
            layers (list of tuples): Configuration of the convolutional layers.
            prefix (str): Prefix to use for naming the layers.

        Returns:
            Sequential: A sequence of convolutional layers.
        """
        conv_layers = OrderedDict()
        for index, (
            in_channel,
            out_channel,
            kernel_size,
            stride,
            padding,
            pool_kernel,
            pool_stride,
            dropout,
        ) in enumerate(layers):
            conv_layers[f"{prefix}_conv_{index}"] = nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            conv_layers[f"{prefix}_conv_act_{index}"] = nn.ReLU()
            conv_layers[f"{prefix}_max_pool_{index}"] = nn.MaxPool2d(
                kernel_size=pool_kernel, stride=pool_stride
            )
            conv_layers[f"{prefix}_dropout_{index}"] = nn.Dropout(p=dropout)

        return nn.Sequential(conv_layers)

    def make_fc_layers(self, layers, prefix):
        """
        Creates a series of fully connected layers.

        Args:
            layers (list of tuples): Configuration of the fully connected layers.
            prefix (str): Prefix to use for naming the layers.

        Returns:
            Sequential: A sequence of fully connected layers.
        """
        fc_layers = OrderedDict()
        for index, (in_feature, out_feature, dropout) in enumerate(layers[:-1]):
            fc_layers[f"{prefix}_{index}"] = nn.Linear(
                in_features=in_feature, out_features=out_feature
            )
            fc_layers[f"{prefix}_act_{index}"] = nn.ReLU()
            fc_layers[f"{prefix}_drop_{index}"] = nn.Dropout(p=dropout)

        in_feature, out_feature = layers[-1]
        fc_layers[f"{prefix}_output"] = nn.Linear(
            in_features=in_feature, out_features=out_feature
        )
        fc_layers[f"{prefix}_output_act"] = nn.Softmax(dim=1)
        return nn.Sequential(fc_layers)

    # Example modification in the combined layer definition
    def make_combined_layer(self, layers, prefix):
        combined_layers = OrderedDict()
        for index, (in_channel, out_channel) in enumerate(layers):
            combined_layers[f"{prefix}_combined_{index}"] = nn.Linear(
                in_features=in_channel, out_features=out_channel
            )
            combined_layers[f"{prefix}_combined_act_{index}"] = nn.ReLU(
                inplace=False  # Set inplace to False
            )
        return nn.Sequential(combined_layers)

    # Make similar changes in other parts of the model where LeakyReLU is used

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            tuple: Outputs from the three branches.
        """
        left = self.left_conv(x)
        middle = self.middle_conv(x)
        right = self.right_conv(x)

        left = left.view(left.size(0), -1)
        middle = middle.view(middle.size(0), -1)
        right = right.view(right.size(0), -1)

        concat = torch.cat((left, middle, right), dim=1)

        combined = self.combined_layer(concat)

        output = self.left_fc(combined)

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model defined".title())

    parser.add_argument(
        "--model", help="Model is defined to classify the disease".capitalize()
    )

    args = parser.parse_args()

    if args.model:
        logging.info("model is calling".capitalize())
        model = Classifier()

        logging.info("model's trainable parameters is calculating".capitalize())
        total_parameters = 0
        for layer_name, parameters in model.named_parameters():
            print(
                "{} & trainable parameters # {} ".format(layer_name, parameters.numel())
            )
            total_parameters = total_parameters + parameters.numel()

        print("Total trainable parameters: {}".format(total_parameters).upper())

        logging.info("Model is defined".capitalize())
    else:
        logging.error("Model is not defined".capitalize())
