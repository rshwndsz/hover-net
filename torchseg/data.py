# Python STL
import os
import logging
from typing import Dict
# Image Processing
import cv2
# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset
# Data augmentation
from albumentations.augmentations import transforms as tf
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

# Root folder of dataset
_DIRNAME: str = os.path.dirname(__file__)
DATA_FOLDER: str = os.path.join(_DIRNAME, "dataset/raw/")


# TODO: Generalize binary segmentation to multiclass segmentation
class OrganDataset(Dataset):
    def __init__(self,
                 data_folder: str,
                 phase: str,
                 cli_args,
                 num_classes: int = 2,
                 class_dict: Dict[int, int] = (0, 255),):
        """Create an API for the dataset

        Parameters
        ----------
        data_folder : str
            Root folder of dataset
        phase : str
            Phase of learning
            In ['train', 'val']
        num_classes : int
            Number of classes (including background)
        class_dict : dict[int, int]
            Dictionary mapping brightness to class indices
        cli_args
            Command line arguments provided at the CLI of `main.py`
        """

        logger = logging.getLogger(__name__)
        logger.info(f"Creating {phase} dataset")

        # Root folder of the dataset
        if not os.path.isdir(data_folder):
            raise NotADirectoryError(f"{data_folder} is not a directory or "
                                     f"it doesn't exist.")
        logger.info(f"Datafolder: {data_folder}")
        self.root = data_folder

        # Phase of learning
        if phase not in ['train', 'val']:
            raise ValueError("Provide any one of ['train', 'val'] as phase.")
        logger.info(f"Phase: {phase}")
        self.phase = phase

        # Data Augmentations and tensor transformations
        self.transforms = OrganDataset.get_transforms(self.phase, cli_args)

        # Get names & number of images in root/train or root/val
        _path_to_imgs = os.path.join(self.root, self.phase, "imgs")
        assert os.path.isdir(_path_to_imgs), f"{_path_to_imgs} doesn't exist."
        self.image_names = sorted(os.listdir(_path_to_imgs))
        assert len(self.image_names) != 0, f"No images found in {_path_to_imgs}"
        logger.info(f"Found {len(self.image_names)} images in {_path_to_imgs}")

        # Number of classes in the segmentation target
        if not isinstance(num_classes, int):
            raise TypeError("Number of classes must be an integer.")
        if not num_classes >= 2:
            raise ValueError(f"Number of classes must be >= 2. "
                             f"2: Binary, >2: Multi-class")
        self.num_classes = num_classes

        # Dictionary specifying the mapping
        # between pixel values [0, 255] and class indices [0, C-1]
        if not len(class_dict) == self.num_classes:
            raise ValueError(f"Length of class dict must be same "
                             f"as number of classes.")
        if not max(class_dict) == 255:
            raise ValueError(f"Max intensity of grayscale images is 255, but "
                             f"class dict: {class_dict} specifies otherwise")
        if not min(class_dict) == 0:
            raise ValueError(f"Min intensity of grayscale images is 0, but "
                             f"class dict: {class_dict} specifies otherwise")
        self.class_dict = class_dict

        # CLI args
        self.cli_args = cli_args

    def __getitem__(self, idx: int):
        # Get image filename
        image_name = self.image_names[idx]
        # Construct path to image from filename
        image_path = os.path.join(self.root, self.phase, "imgs", image_name)
        # Read image with opencv
        if self.cli_args.in_channels == 1:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(image_path)
        # Check if image has been read properly
        if image.size == 0:
            raise IOError(f"cv2: Unable to load image - {image_path}")

        # Mask has the same filename as image
        mask_name = image_name
        # Construct path to mask
        mask_path = os.path.join(self.root, self.phase, "masks", mask_name)
        # Masks should have int values in [0, C-1] where C => Number of classes
        # Checking the above condition for every mask is extremely inefficient
        # So it's left to you 🙇
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Check if mask has been read properly
        if mask.size == 0:
            raise IOError(f"cv2: Unable to load mask - {mask_path}")

        # Data Augmentation for image and mask
        augmented = self.transforms['aug'](image=image, mask=mask)
        new_image = self.transforms['img_only'](image=augmented['image'])
        new_mask = self.transforms['mask_only'](image=augmented['mask'])
        aug_tensors = self.transforms['final'](image=new_image['image'],
                                               mask=new_mask['image'])
        image = aug_tensors['image']
        mask = aug_tensors['mask']

        # Add a channel dimension (C in [N C H W]) if required
        if self.num_classes == 2:
            mask = torch.unsqueeze(mask, dim=0)  # [H, W] => [H, W]

        # Return tuple of tensors
        return image, mask

    def __len__(self):
        return len(self.image_names)

    @staticmethod
    def get_transforms(phase: str, cli_args) -> Dict[str, Compose]:
        """Get composed albumentations augmentations

        Parameters
        ----------
        phase : str
            Phase of learning
            In ['train', 'val']
        cli_args
            Arguments coming all the way from `main.py`

        Returns
        -------
        transforms: dict[str, albumentations.core.composition.Compose]
            Composed list of transforms
        """
        aug_transforms = []
        im_sz = (cli_args.image_size, cli_args.image_size)

        if phase == "train":
            # Data augmentation for training only
            aug_transforms.extend([
                tf.ShiftScaleRotate(
                    shift_limit=0,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5),
                tf.Flip(p=0.5),
                tf.RandomRotate90(p=0.5),
            ])
            # Exotic Augmentations for train only 🤤
            aug_transforms.extend([
                tf.RandomBrightnessContrast(p=0.5),
                tf.ElasticTransform(p=0.5),
                tf.MultiplicativeNoise(multiplier=(0.5, 1.5),
                                       per_channel=True, p=0.2),
            ])
        aug_transforms.extend([
            tf.RandomSizedCrop(min_max_height=im_sz,
                               height=im_sz[0],
                               width=im_sz[1],
                               w2h_ratio=1.0,
                               interpolation=cv2.INTER_LINEAR,
                               p=1.0),
        ])
        aug_transforms = Compose(aug_transforms)

        mask_only_transforms = Compose([
            tf.Normalize(mean=0, std=1, always_apply=True)
        ])
        image_only_transforms = Compose([
            tf.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0),
                         always_apply=True)
        ])
        final_transforms = Compose([
            ToTensorV2()
        ])

        transforms = {
            'aug': aug_transforms,
            'img_only': image_only_transforms,
            'mask_only': mask_only_transforms,
            'final': final_transforms
        }
        return transforms


def provider(data_folder: str,
             phase: str,
             cli_args,
             batch_size: int = 8,
             num_workers: int = 4,) -> object:
    """Return dataloader for the dataset

    Parameters
    ----------
    data_folder : str
        Root folder of the dataset
    phase : str
        Phase of learning
        In ['train', 'val']
    batch_size : int
        Batch size
    num_workers : int
        Number of workers
    cli_args
        Command line arguments provided at the CLI of `main.py`

    Returns
    -------
    dataloader: torch.utils.data.DataLoader
        DataLoader for loading data from CPU to GPU
    """
    image_dataset = OrganDataset(data_folder, phase, cli_args)
    logger = logging.getLogger(__name__)
    logger.info(f"Creating {phase} dataloader")
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    return dataloader
