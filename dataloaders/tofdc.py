import numpy as np
import os
import random
import cv2
import torch

from torchvision.transforms import Normalize
from torch.utils.data import Dataset
from PIL import Image


def list_from_file(split: str):
    assert split in ['train', 'test']

    with open(os.path.join('data_splits/TOFDC/', f'TOFDC_{split}.txt'), 'r') as f:
        lines = f.readlines()

    samples = []
    for line in lines:
        rgb_path, gt_path, _ = line.split(',')
        samples.append(
            (rgb_path, gt_path)
        )

    return samples


class TOFDCDataset(Dataset):
    def __init__(self, root_dir: str, split: str, max_translation_x: int, max_rotate_degree: float):
        assert split in ['train', 'test']

        self.root_dir = root_dir
        self.split = split

        self.max_translation_x = max_translation_x
        self.max_rotate_degree = max_rotate_degree

        # read samples
        self.samples = list_from_file(split)

        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, dpt_path = self.samples[idx]

        # read image
        image = Image.open(os.path.join(self.root_dir, rgb_path))
        # read depth
        depth_gt = Image.open(os.path.join(self.root_dir, dpt_path))

        # Random rotate
        if self.split == 'train':
            random_angle = (random.random() - 0.5) * 2 * self.max_rotate_degree
            image = self.rotate_image(image, random_angle)
            depth_gt = self.rotate_image(depth_gt, random_angle, Image.NEAREST)

        # convert to np.ndarray
        image = np.asarray(image, dtype=np.uint8)
        depth_gt = np.asarray(depth_gt, dtype=np.float32) / 1000.0
        depth_gt = np.expand_dims(depth_gt, axis=2)

        # filter depth_gt
        depth_gt[(depth_gt < 0.1) | (depth_gt > 5.0)] = 0.

        if self.split == 'train':
            # augment image
            image_aug = image.copy()
            image_aug = self.random_color_augment(image_aug)
            image_aug = np.clip(image_aug, 0, 255).astype(np.float32) / 255.0
            image = image.astype(np.float32) / 255.0

            image, image_aug, depth_gt = self.train_preprocess(image, image_aug, depth_gt)
            image, image_aug, depth_gt = self.translateX(image, image_aug, depth_gt)

            # transform
            image = self.normalize(self.to_tensor(image))
            image_aug = self.normalize(self.to_tensor(image_aug))
            depth_gt = self.to_tensor(depth_gt)

            sample = {'image': image, 'image_aug': image_aug, 'depth': depth_gt}
        else:
            image = image.astype(np.float32) / 255.0

            image = self.normalize(self.to_tensor(image))
            depth_gt = self.to_tensor(depth_gt)

            sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': True}

        return sample

    @staticmethod
    def to_tensor(img: np.ndarray):
        return torch.from_numpy(img.transpose((2, 0, 1)))

    @staticmethod
    def rotate_image(image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def translateX(self, img: np.ndarray, img_aug: np.ndarray, gt: np.ndarray):
        def translate(x: np.ndarray, t: int):
            if t >= 0:
                pad = ((0, 0), (t, 0), (0, 0))
                x = np.pad(x, pad)
                return x[:, : W, :]
            else:
                pad = ((0, 0), (0, -t), (0, 0))
                x = np.pad(x, pad)
                return x[:, -W:, :]

        if random.random() > 0.5:
            return img, img_aug, gt

        H, W, C = img.shape
        max_tx = self.max_translation_x
        candidates = list(range(-(max_tx - 1), max_tx))
        tx = random.choice(candidates)
        return translate(img, tx), translate(img_aug, tx), translate(gt, tx)

    def train_preprocess(self, image, image_aug, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            image_aug = (image_aug[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image_aug = self.augment_image(image_aug)

        return image, image_aug, depth_gt

    @staticmethod
    def augment_image(image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    @staticmethod
    def random_color_augment(img: np.ndarray):
        if random.random() > 0.5:
            return img

        candidates = {
            'mean_blur': 0.1,
            'median_blur': 0.1,
            'gaussian_blur': 0.1,
            # 'equalize',
            'uniform_noise': 0.2,
            'gaussian_noise': 0.2,
            'sharpen': 0.3
        }

        augment = np.random.choice(
            list(candidates.keys()),
            1,
            p=list(candidates.values())
        )

        if augment == 'mean_blur':
            k_size = random.randint(3, 9)
            return cv2.blur(img, (k_size, k_size))
        elif augment == 'median_blur':
            k_size = random.choice([3, 5, 7, 9])
            return cv2.medianBlur(img, k_size)
        elif augment == 'gaussian_blur':
            k_size = random.choice([3, 5, 7, 9])
            sigma = random.random() * 10.
            return cv2.GaussianBlur(img, (k_size, k_size), sigma)
        elif augment == 'equalize':
            return np.stack(
                [
                    cv2.equalizeHist(img[:, :, idx])
                    for idx in range(3)
                ], -1
            )
        elif augment == 'uniform_noise':
            scale = random.random() * 50.
            noise = np.random.uniform(-scale, scale, img.shape)
            return img.astype(np.float32) + noise
        elif augment == 'gaussian_noise':
            scale = random.random() * 40.
            noise = np.random.normal(0., scale, img.shape)
            return img.astype(np.float32) + noise
        elif augment == 'sharpen':
            identity = np.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]
                ], dtype=np.float32)
            sharpen = np.array(
                [
                    [0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]
                ], dtype=np.float32) / 4.
            sharp = sharpen * random.random() * 4.0
            kernel = identity + sharp
            return cv2.filter2D(img, -1, kernel)
        else:
            raise NotImplementedError
