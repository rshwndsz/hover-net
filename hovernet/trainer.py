# Python STL
import os
import logging
from typing import Dict, Tuple
# PyTorch
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.backends.cudnn as cudnn

# Local
from .loss import MixedLoss
from .data import provider
from .data import DATA_FOLDER
from hovernet.storage import Meter

_DIRNAME = os.path.dirname(__file__)


class Trainer(object):
    """An object to encompass all training and validation

    Training loop, validation loop, logging, checkpoints are all
    implemented here.

    Attributes
    ----------
    num_workers : int
        Number of workers
    batch_size : int
        Batch size
    lr : int
        Learning rate
    num_epochs : int
        Number of epochs
    current_epoch : int
        Current epoch
    phases : list[str]
        List of learning phases
    val_freq : int
        Validation frequency
    device : torch.device
        GPU or CPU
    checkpoint_path : str
        Path to checkpoint file
    save_path : str
        Path to file where state will be saved
    net
        Our NN in PyTorch
    criterion
        Loss function
    optimizer
        Optimizer
    scheduler
        Learning rate scheduler
    dataloaders : dict[str, torch.utils.data.DataLoader]
        Dataloaders for each phase
    best_loss : float
        Best validation loss
    meter : Meter
        Object to store loss & scores
    """
    def __init__(self, model, args):
        """Initialize a Trainer object

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model of your NN
        args : :obj:
            CLI arguments
        """

        # Set hyperparameters
        self.num_workers: int = args.num_workers
        self.batch_size: Dict[str, int] = {"train": args.batch_size,
                                           "val": args.batch_size}
        self.lr: float = args.lr
        self.num_epochs: int = args.num_epochs
        self.current_epoch: int = 0
        self.phases: Tuple[str, ...] = ("train", "val")
        self.val_freq: int = args.val_freq

        # Torch-specific initializations
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")
        else:
            self.device = torch.device("cuda:0")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

        if args.checkpoint_name is not None:
            self.checkpoint_path: str = os.path.join(_DIRNAME, "checkpoints",
                                                     args.checkpoint_name)
        else:
            self.checkpoint_path = None

        # Path where best model will be saved
        self.save_path: str = os.path.join(_DIRNAME, "checkpoints",
                                           args.save_fname)

        # Model, loss, optimizer & scheduler
        self.net = model
        # <<<< Catch: https://pytorch.org/docs/stable/optim.html
        self.net = self.net.to(self.device)
        self.criterion = MixedLoss(9.0, 4.0)
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",
                                           patience=3, verbose=True,
                                           cooldown=0, min_lr=3e-6)

        # Faster convolutions at the expense of memory
        cudnn.benchmark = True

        # Get loaders for training and validation
        self.dataloaders = {
            phase: provider(
                data_folder=DATA_FOLDER,
                phase=phase,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
                cli_args=args
            )
            for phase in self.phases
        }

        # Initialize losses & scores
        self.best_loss: float = float("inf")  # Very high
        self.meter = Meter(self.phases, scores=('loss', 'iou', 'dice',
                                                'acc', 'prec'))

    def forward(self,
                images: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass

        Parameters
        ----------
        images : torch.Tensor
            Input to the NN
        targets : torch.Tensor
            Supervised labels for the NN

        Returns
        -------
        loss: torch.Tensor
            Loss from one forward pass
        logits: torch.Tensor
            Raw output of the NN, without any activation function
            in the last layer
        """

        images: torch.Tensor = images.to(self.device)
        masks: torch.Tensor = targets.to(self.device)
        logits: torch.Tensor = self.net(images)
        loss: torch.Tensor = self.criterion(logits, masks)
        return loss, logits

    def iterate(self,
                epoch: int,
                phase: str) -> float:
        """1 epoch in the life of a model

        Parameters
        ----------
        epoch : int
            Current epoch
        phase : str
            Phase of learning
            In ['train', 'val']
        Returns
        -------
        epoch_loss: float
            Average loss for the epoch
        """

        # Set model & dataloader based on phase
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]

        # ===ON_EPOCH_BEGIN===
        self.meter.on_epoch_begin(epoch, phase)

        # Learning!
        self.optimizer.zero_grad()

        # TODO: Add progress bar
        for itr, batch in enumerate(dataloader):
            # Load images and targets
            images, targets = batch

            # ===ON_BATCH_BEGIN===
            self.meter.on_batch_begin()

            # Forward pass
            loss, logits = self.forward(images, targets)
            if phase == "train":
                # Backprop for training only
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # Update metrics for this batch
            with torch.no_grad():
                loss = loss.detach().cpu()
                logits = logits.detach().cpu()

                # ===ON_BATCH_CLOSE===
                self.meter.on_batch_close(loss=loss,
                                          logits=logits, targets=targets)

        # ===ON_EPOCH_CLOSE===
        # Collect loss & scores
        self.meter.on_epoch_close()

        # Empty GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return average loss from the criterion for this epoch
        return self.meter.store['loss'][phase][-1]

    def start(self):
        """Start the loops!"""

        # ===ON_TRAIN_BEGIN===
        self.meter.on_train_begin()

        # <<< Change: Hardcoded starting epoch
        for epoch in range(1, self.num_epochs + 1):
            # Update current_epoch
            self.current_epoch: int = epoch

            # Train model for 1 epoch
            self.iterate(epoch, "train")

            # Construct the state for a possible save later
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            # Validate model for `val_freq` epochs
            if epoch % self.val_freq == 0:
                val_loss = self.iterate(epoch, "val")

                # Step the scheduler based on validation loss
                self.scheduler.step(val_loss)

                # Save model if val loss is lesser than anything seen before
                if val_loss < self.best_loss:
                    logger = logging.getLogger(__name__)
                    logger.info("****** New optimal found, saving state ******")
                    state["best_loss"] = self.best_loss = val_loss
                    try:
                        torch.save(state, self.save_path)
                    except FileNotFoundError:
                        logger.exception(f"Error while saving checkpoint",
                                         exc_info=True)

            # Print newline
            print()

        # ===ON_TRAIN_END===
        self.meter.on_train_close()
