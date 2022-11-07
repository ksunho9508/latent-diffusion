import os, yaml, pickle, shutil, tarfile, glob
import pandas as pd
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset

import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light


class FundusSR(Dataset):
    def __init__(self, data_dir=None, size=None,
                 degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
                 random_crop=True):
        """
        Imagenet Superresolution Dataloader
        Performs following ops in order:
        1.  crops a crop of size s from image either as random or center crop
        2.  resizes crop to size with cv2.area_interpolation
        3.  degrades resized crop with degradation_fn

        :param size: resizing to size after cropping
        :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
        :param downscale_f: Low Resolution Downsample factor
        :param min_crop_f: determines crop size s,
          where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
        :param max_crop_f: ""
        :param data_root:
        :param random_crop:
        """
        self.data_dir = data_dir
        self.base = self.get_base()
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow
 
        interpolation_fn = {
        "cv_nearest": cv2.INTER_NEAREST,
        "cv_bilinear": cv2.INTER_LINEAR,
        "cv_bicubic": cv2.INTER_CUBIC,
        "cv_area": cv2.INTER_AREA,
        "cv_lanczos": cv2.INTER_LANCZOS4,
        "pil_nearest": PIL.Image.NEAREST,
        "pil_bilinear": PIL.Image.BILINEAR,
        "pil_bicubic": PIL.Image.BICUBIC,
        "pil_box": PIL.Image.BOX,
        "pil_hamming": PIL.Image.HAMMING,
        "pil_lanczos": PIL.Image.LANCZOS,
        }[degradation]

        # self.pil_interpolation = degradation.startswith("pil_")

        self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                        interpolation=interpolation_fn)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = self.base[i]
        image = Image.open(example)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = np.array(image).astype(np.uint8)

        min_side_len = min(image.shape[:2])
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)

        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)

        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image=image)["image"]
  
        LR_image = self.degradation_process(image=image)["image"]
 

        return {'image': (image/127.5 - 1.0).astype(np.float32),
                'LR_image': (LR_image/127.5 - 1.0).astype(np.float32)}
    
    # def get_iter_per_epoch(self):
    #     return len(self) // (self.batch_size * self.devices)

class FundusSRTrain(FundusSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        csv_dir = os.path.join(self.data_dir, 'label_files/finding_train.csv')
        df = pd.read_csv(csv_dir)
        df["filename"] = df["filename"].apply(lambda x: x.replace("/media/ext", self.data_dir))
        return df["filename"] 

class FundusSRValidation(FundusSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        csv_dir = os.path.join(self.data_dir, 'label_files/finding_val.csv')
        df = pd.read_csv(csv_dir)
        df["filename"] = df["filename"].apply(lambda x: x.replace("/media/ext", '/data1/fundus_dataset/inhouse_dataset'))
        return df["filename"]

class FundusSRTest(FundusSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_base(self):
        csv_dir = os.path.join(self.data_dir, 'label_files/finding_test.csv')
        df = pd.read_csv(csv_dir)
        df["filename"] = df["filename"].apply(lambda x: x.replace("/media/ext", '/data1/fundus_dataset/inhouse_dataset'))
        return df["filename"]