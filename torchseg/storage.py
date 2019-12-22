# Python STL
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
# PyTorch
import torch

# Local
from torchseg import utils
from torchseg import metrics


# TODO: Move to an event system
# TODO: Have a better way to add metrics & associated functions
# See: https://stackoverflow.com/questions/1092531/event-system-in-python/

class Meter(object):
    def __init__(self,
                 phases: Tuple[str, ...] = ('train', 'val'),
                 scores: Tuple[str, ...] = ('loss', 'iou')):

        self.phases: Tuple[str, ...] = phases
        self.current_phase: str = 'train'
        self.scores: Tuple[str, ...] = scores
        self.store: Dict[str, Dict[str, List[float]]] = {
            score: {
                phase: [] for phase in self.phases
            } for score in self.scores
        }
        self.base_threshold: float = 0.5
        self.metrics: Dict[str, List] = {
            score: [] for score in self.scores
        }

        self.epoch_start_time: datetime = datetime.now()

    def on_train_begin(self):
        pass

    def on_epoch_begin(self,
                       current_epoch: int,
                       current_phase: str,):

        # Log epoch, phase and start time
        self.epoch_start_time: datetime = datetime.now()
        epoch_start_time_string = datetime.strftime(self.epoch_start_time,
                                                    '%I:%M:%S %p')
        logger = logging.getLogger(__name__)
        logger.info(f"Starting epoch: {current_epoch} | "
                    f"phase: {current_phase} | "
                    f"@ {epoch_start_time_string}")

        # Initialize metrics
        self.metrics: Dict[str, List] = {
            score: [] for score in self.scores
        }

        # For later
        self.current_phase: str = current_phase

    def on_batch_begin(self):
        pass

    def on_batch_close(self,
                       loss: torch.Tensor,
                       logits: torch.Tensor,
                       targets: torch.Tensor,):

        # Get predictions and probabilities from raw logits
        probs: torch.Tensor = torch.sigmoid(logits)
        preds: torch.Tensor = utils.predict(probs, self.base_threshold)

        # Assertion for shapes
        if not (preds.shape == targets.shape):
            raise ValueError(f"Shape of preds: {preds.shape} must be the same "
                             f"as that of targets: {targets.shape}.")

        # Add loss to list
        self.metrics['loss'].append(loss)

        # Calculate and add to metric lists
        dice: torch.Tensor = metrics.dice_score(probs, targets,
                                                self.base_threshold)
        self.metrics['dice'].append(dice)

        iou: torch.Tensor = metrics.iou_score(preds, targets)
        self.metrics['iou'].append(iou)

        acc: torch.Tensor = metrics.accuracy_score(preds, targets)
        self.metrics['acc'].append(acc)

        # <<< Change: Hardcoded for binary segmentation
        prec: torch.Tensor = metrics.precision_score(preds, targets)[1]
        self.metrics['prec'].append(prec)

    def on_epoch_close(self):

        # Average over metrics obtained for every batch in the current epoch
        self.metrics.update({key: [
            utils.nanmean(torch.tensor(self.metrics[key])).item()
        ] for key in self.metrics.keys()})

        # Compute time taken to complete the epoch
        epoch_end_time: datetime = datetime.now()
        delta_t: timedelta = epoch_end_time - self.epoch_start_time

        # Construct string for logging
        metric_string: str = f""
        for metric_name, metric_value in self.metrics.items():
            metric_string += f"{metric_name}: {metric_value[0]:.4f} | "
        metric_string += f"in {delta_t.seconds}s"

        # Log metrics & time taken
        logger = logging.getLogger(__name__)
        logger.info(f"{metric_string}")

        # Put metrics for this epoch in long term (complete training) storage
        for s in self.store.keys():
            try:
                self.store[s][self.current_phase].extend(self.metrics[s])
            except KeyError:
                logger = logging.getLogger(__name__)
                logger.warning(f"Key '{s}' not found. Skipping...",
                               exc_info=True)
                continue

    def on_train_close(self):
        pass
