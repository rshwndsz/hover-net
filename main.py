# Python STL
import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
from logging.config import dictConfig
# Data Science
import matplotlib.pyplot as plt
# PyTorch
import torch
# Fancy logs
import coloredlogs

# Local
from torchseg.model import model
from torchseg.trainer import Trainer

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


def cli():
    parser = ArgumentParser(description='Torchseg',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # Pre-training based
    parser.add_argument('-c', '--checkpoint', dest='checkpoint_name', type=str,
                        nargs="?", default=None, const="model.pth",
                        help='Name of checkpoint file in torchseg/checkpoints/')
    parser.add_argument('-s', '--save', dest='save_fname', type=str,
                        default="model-saved.pth",
                        help="File in checkpoints/ to save state")
    parser.add_argument('-p', '--pretrained', dest='pretrained',
                        action="store_true",
                        help="Load imagenet weights or not")
    parser.set_defaults(feature=True)
    # Training loop based
    parser.add_argument('-b', '--batch_size', dest="batch_size", type=int,
                        default=32,
                        help='Batch size')
    parser.add_argument('-w', '--workers', dest='num_workers', type=int,
                        default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--lr', dest='lr', type=float,
                        default=3e-4,
                        help='Initial learning rate')
    parser.add_argument('-e', '--epoch', dest='num_epochs', type=int,
                        default=10,
                        help='Number of epochs')
    parser.add_argument('-v', '--val_freq', dest='val_freq', type=int,
                        default=5,
                        help='Validation frequency')
    # Image based
    parser.add_argument('--image_size', dest='image_size', type=int,
                        default=256, help='Resize images to size: s*s')
    parser.add_argument('--in_channels', dest='in_channels', type=int,
                        default=3,
                        help='Number of channels in input image')

    parser_args = parser.parse_args()

    # Some checks on provided args
    if parser_args.checkpoint_name is not None:
        _checkpoint_path = os.path.join(_DIRNAME, "torchseg", "checkpoints",
                                        parser_args.checkpoint_name)
        if not os.path.exists(_checkpoint_path):
            raise FileNotFoundError("The checkpoints file at {} was not found."
                                    "Check the name again."
                                    .format(_checkpoint_path))
        else:
            logger.info(f"Loading checkpoint file: {_checkpoint_path}")
    else:
        logger.info(f"Training from scratch")

    if parser_args.pretrained:
        logger.info("Using pretrained model")
    else:
        logger.info("Using random weights")

    if parser_args.num_epochs <= 0:
        raise ValueError("Number of epochs: {} must be > 0"
                         .format(parser_args.num_epochs))
    else:
        logger.info(f"Number of epochs: {parser_args.num_epochs}")

    if parser_args.num_workers < 0:
        raise ValueError("Number of workers: {} must be >= 0"
                         .format(parser_args.num_workers))
    else:
        logger.info(f"Number of workers: {parser_args.num_workers}")

    if parser_args.lr < 0:
        raise ValueError("Learning rate: must be >= 0"
                         .format(parser_args.lr))
    else:
        logger.info(f"Initial Learning rate: {parser_args.lr}")

    if (parser_args.val_freq <= 0 or
            parser_args.val_freq > parser_args.num_epochs):
        raise ValueError("Validation frequency: {} must be > 0 and "
                         "less than number of epochs"
                         .format(parser_args.val_freq))
    else:
        logger.info(f"Validation frequency: {parser_args.val_freq} epochs")

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
    # CLI
    args = cli()

    # Get trainer
    model_trainer = Trainer(model, args)
    # Save model before exiting if there is a keyboard interrupt
    try:
        # Start training + validation
        model_trainer.start()
    except KeyboardInterrupt or SystemExit as e:
        logger.info("Exit requested during train-val")
        # Collect state
        state = {
            "epoch": model_trainer.current_epoch,
            "best_loss": model_trainer.best_loss,
            "state_dict": model_trainer.net.state_dict(),
            "optimizer": model_trainer.optimizer.state_dict(),
        }
        logger.info("******** Saving state before exiting ********")
        # Save state if possible
        try:
            torch.save(state, os.path.join(_DIRNAME, "torchseg",
                                           "checkpoints", args.save_fname))
        except FileNotFoundError as e:
            logger.exception(f"Error while saving checkpoint", exc_info=True)
        else:
            logger.info("Saved 🎉")

        sys.exit(0)

    # Helper function to plot scores
    def metric_plot(scores, name):
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(scores["train"])),
                 scores["train"],
                 label=f'train {name}')
        plt.plot(range(len(scores["val"])),
                 scores["val"],
                 label=f'val {name}')
        plt.title(f'{name} plot')
        plt.xlabel('Epoch')
        plt.ylabel(f'{name}')
        plt.legend()
        plt.show()

    for metric_name, metric_values in model_trainer.meter.store.items():
        metric_plot(metric_values, metric_name)
