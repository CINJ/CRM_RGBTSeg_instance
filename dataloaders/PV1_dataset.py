# Written by Ukcheol Shin, Jan. 24, 2023 using the following two repositories.
# MS-UDA: https://github.com/yeong5366/MS-UDA
# Mask2Former: https://github.com/facebookresearch/Mask2Former

import cv2
import numpy as np
import os, torch
from imageio import imread
from skimage.transform import resize
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from detectron2.structures import BitMasks, Instances
from detectron2.data import transforms as T
from .augmentation import ColorAugSSDTransform, MaskGenerator


def numerical_sort(value):
    return "".join([o for o in value if o.isnumeric()])
    

class PV1_dataset(Dataset):
    """
    Single class instance segmentation
    """
    def __init__(self, data_dir, cfg, split):
        super(PV1_dataset, self).__init__()
        split = split.strip().lower()
        self.resize_dims = cfg.INPUT.CROP.SIZE
        self.split = split if split != "val" else "test"
        self.data_dir = os.path.join(data_dir, self.split)
        self.data_list = []
        for pack_name in sorted(os.listdir(self.data_dir), key=numerical_sort):
            if os.path.isdir(os.path.join(self.data_dir, pack_name)):
                self.data_list.append(pack_name)
        self.n_data = len(self.data_list)
        self.size_divisibility = -1
        self.ignore_label = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.augmentations = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            self.augmentations.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            self.augmentations.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        self.augmentations.append(T.RandomFlip())
        if cfg.INPUT.MASK.ENABLED:
            self.mask_generator = MaskGenerator(input_size=cfg.INPUT.MASK.SIZE, \
                                                mask_patch_size=cfg.INPUT.MASK.PATCH_SIZE, \
                                                model_patch_size=cfg.MODEL.SWIN.PATCH_SIZE, \
                                                mask_ratio=cfg.INPUT.MASK.RATIO,
                                                mask_type=cfg.INPUT.MASK.TYPE,
                                                strategy=cfg.INPUT.MASK.STRATEGY
                                                )
        else:
            self.mask_generator = None 

    def __read_image(self, filename):
        src = imread(filename).astype('float32') # HxWxC
        return resize(src, self.resize_dims , mode='reflect', anti_aliasing=True)
  
    def __image_to_binary_mask(self, source_image):
        instance_masks_grayscale = torch.mean(torch.from_numpy(source_image), dim=-1)
        # Step 2: Apply a threshold to convert to binary (0 or 1)
        # Assuming any non-zero value in the grayscale mask represents the label
        return (instance_masks_grayscale > 0).float()
    
    def __merge_instance_masks(self, source_rgb_image, instance_masks=[]):
        """
        Optional to send list of instance masks, otherwise you get back an empty mask 
        of the original image shape.
        """
        label_map = np.zeros_like(torch.mean(torch.from_numpy(source_rgb_image), dim=-1), dtype=np.float32)
        for instance_mask in instance_masks:
            label_map[instance_mask > 0] = 1  # Set the class label where the mask is present
        return label_map
    
    def __getitem__(self, index):
        # filename mapping
        pack_name  = self.data_list[index]
        base_path = os.path.join(self.data_dir, pack_name)
        rgb_name = os.path.join(base_path, "rgb.jpg")
        lwir_name = os.path.join(base_path, "lwir.jpg")
        mask_names = [os.path.join(base_path, filename) for filename in sorted(os.listdir(base_path), key=numerical_sort) if filename.startswith("mask")]

        # load images
        image_rgb = self.__read_image(rgb_name)
        image_thr = np.expand_dims(self.__read_image(lwir_name).mean(axis=2), axis=2)
        image = np.concatenate((image_rgb,image_thr),axis=2)
        instance_masks: [] = [self.__image_to_binary_mask(self.__read_image(mask_name)) for mask_name in mask_names]
        sem_seg_gt = self.__merge_instance_masks(image_rgb, instance_masks)

        # Pad image
        image      = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        image_shape = (image.shape[-2], image.shape[-1])  # h, w
        instances = Instances(image_shape)
        # TODO: change this to be a zero list (of size one)
        # print(f"Number of masks being stacked: {len(instance_masks)}")
        if len(instance_masks) > 0:
            # modified to include a background (inverted) mask with label 0
            instances.gt_classes = torch.cat([torch.ones(len(instance_masks), dtype=torch.int64), torch.tensor([0], dtype=torch.int64)], dim=0)
            bgmask = torch.from_numpy(~(self.__merge_instance_masks(image_rgb, instance_masks).astype(np.uint8)))
            mask_tensor = torch.stack([torch.from_numpy(np.ascontiguousarray(mask)) for mask in instance_masks])
            mask_tensor = torch.cat([mask_tensor, bgmask.unsqueeze(0)], dim=0)    
            # print(f"mask_tensor dim: {mask_tensor.dim()} mask_tensor size: {mask_tensor.size()} mask_tensor.shape: {mask_tensor.shape}")
            instances.gt_masks = BitMasks(mask_tensor).tensor
        else:
            # only provides a background mask with label 0
            instances.gt_classes = torch.zeros(1, dtype=torch.int64)
            instances.gt_masks = BitMasks(torch.zeros(image_shape, dtype=torch.uint8).unsqueeze(0)).tensor

        # Pack payload
        result = {}
        result["name"]  = pack_name
        result["image"] = image
        result["sem_seg_gt"] = sem_seg_gt.long()        
        result["instances"] = instances
        if (self.split == 'train') and (self.mask_generator is not None):
            mask1, mask2 = self.mask_generator()
            result["mask"] = torch.as_tensor(np.stack([mask1, mask2], axis=0))
        return result

    def __len__(self):
        return self.n_data
