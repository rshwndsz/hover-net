# Python STL
import os
import sys
import argparse
import logging
from logging.config import dictConfig
from typing import List
# Data Science
import matplotlib.pyplot as plt
# Image processing
import cv2
# PyTorch
import torch
from torch.utils.data import Dataset
# Data Augmentations
from albumentations.augmentations import transforms as tf
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
# Fancy logs
import coloredlogs

# Local
from torchseg.data import DATA_FOLDER
from torchseg.model import model

_DIRNAME = os.path.dirname(__file__)

# Colourful 🌈
C_LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'loggers': {
        '': {
           'level': 'DEBUG',
           'handlers': ['console']
        },
    },
    'formatters': {
        'colored_console': {
            '()': 'coloredlogs.ColoredFormatter',
            'format': "%(asctime)s - %(name)-18s - %(levelname)-8s"
                      " - %(message)s",
            'datefmt': '%H:%M:%S'},
        'format_for_file': {
            'format': "%(asctime)s :: %(levelname)s :: %(funcName)s in "
                      "%(filename)s (l:%(lineno)d) :: %(message)s",
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'colored_console',
            'stream': 'ext://sys.stdout'
        },
    },
}

# Load logging configuration (C_LOGGING => colour, LOGGING_CONFIG => plain)
dictConfig(C_LOGGING)

# Create logger
logger = logging.getLogger(__name__)
# Add colour
coloredlogs.install(fmt=C_LOGGING['formatters']['colored_console']['format'],
                    stream=sys.stdout,
                    level='DEBUG', logger=logger)


class TestDataset(Dataset):
    """API for the test dataset

    Attributes
    ----------
    root : str
        Root folder of the dataset
    image_names : list[str]
        Sorted list of test images
    transform : albumentations.core.composition.Compose
        Albumentations augmentations pipeline
    """
    def __init__(self, data_folder, cli_args):
        self.cli_args = cli_args
        self.root: str = data_folder
        self.image_names: List[str] = sorted(os.listdir(os.path.join(self.root,
                                                                     "test",
                                                                     "imgs")))
        self.transform = Compose(
            [
                # Normalize images to [0..1]
                tf.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), p=1),
                # Resize images to (image_size, image_size)
                tf.Resize(cli_args.image_size, cli_args.image_size),
                # Convert PIL images to torch.Tensor
                ToTensorV2(),
            ]
        )

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Get image name from list of image names
        image_name: str = self.image_names[idx]
        # Construct path to image
        image_path: str = os.path.join(self.root, "test", "imgs", image_name)
        # Read image with opencv
        if self.cli_args.in_channels == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path)

        # Check if image has been read properly
        if image.size == 0:
            raise IOError(f"cv2: Unable to load image - {image_path}")

        # Augment image
        image = self.transform(image=image)["image"]
        # Return augmented image as a tensor
        return image

    def __len__(self):
        return len(self.image_names)


def cli():
    parser = argparse.ArgumentParser(description='Torchseg')
    parser.add_argument('-c', '--checkpoint', dest='checkpoint_name', type=str,
                        default="model-saved.pth",
                        help='Name of checkpoint file in torchseg/checkpoints/')
    parser.add_argument('--image_size', dest='image_size', type=int,
                        default=256, help='Resize images to size: s*s')
    parser.add_argument('--in_channels', dest='in_channels', type=int,
                        default=3,
                        help='Number of channels in input image')

    parser_args = parser.parse_args()

    # Validate provided args
    test_checkpoint_path: str = os.path.join("torchseg", "checkpoints",
                                             parser_args.checkpoint_name)
    if not os.path.exists(test_checkpoint_path):
        raise FileNotFoundError("The checkpoints file at {} was not found. "
                                "Check if the file exists."
                                .format(test_checkpoint_path))
    else:
        logger.info(f"Loading checkpoint file: {test_checkpoint_path}")

    if not parser_args.image_size > 0:
        raise ValueError("Image size must be a positive non-zero integer.")
    else:
        logger.info(f"Images will be resized to "
                    f"({parser_args.image_size}, {parser_args.image_size})")

    if not parser_args.in_channels > 0 and parser_args.in_channels < 16:
        raise ValueError(f"Number of input channels ({parser_args.in_channels})"
                         f" must be within (0, 16)")
    else:
        logger.info(f"Images will be loaded with {parser_args.in_channels} "
                    f"channel{'s' if parser_args.in_channels != 1 else ''}")

    return parser_args


if __name__ == "__main__":

    # Parse CLI arguments
    args = cli()

    # Get test dataset
    testset = TestDataset(DATA_FOLDER, args)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Switch model over to evaluation model
    model.eval()
    # Load state from checkpoint
    checkpoint_path = os.path.join(_DIRNAME, "torchseg",
                                   "checkpoints", args.checkpoint_name)
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["state_dict"])

    # For every image in testset, predict and plot
    with torch.no_grad():
        for i, batch in enumerate(testset):
            batch = batch.unsqueeze(dim=0)
            probs = torch.sigmoid(model(batch.to(device)))
            preds = (probs > 0.5).float()
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(preds.cpu().numpy().squeeze(),
                         'gray')
            ax[1].imshow(batch.cpu().numpy().squeeze().transpose(1, 2, 0),
                         'gray')
            ax[0].set_title("Prediction")
            ax[1].set_title("Image")
            plt.show()
