import os
import random

import cv2
import numpy as np
import torch
import copy
import torch.utils.data.distributed
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None):
        assert mode in ['train', 'online_eval', 'test']

        self.args = args

        filename = {
            'train': args.filenames_file,
            'online_eval': args.filenames_file_eval,
            'test': args.filenames_file_test
        }
        with open(filename[mode], 'r') as f:
            self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = 518.8579

        if self.mode == 'train':
            rgb_file = sample_path.split()[0][11:]
            depth_file = rgb_file.replace('/image_02/data/', '/proj_depth/groundtruth/image_02/')

            if self.args.use_right and random.random() > 0.5:
                rgb_file = rgb_file.replace('image_02', 'image_03')
                depth_file = depth_file.replace('image_02', 'image_03')

            image_path = os.path.join(self.args.data_path, rgb_file)
            depth_path = os.path.join(self.args.gt_path, depth_file)

            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)

            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
            # PIL.Image

            # convert to np.ndarray
            image = np.asarray(image, dtype=np.uint8)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            # # crop again
            # lm = (1216 - self.args.input_width) // 2
            # image = image[-self.args.input_height:, lm: lm + self.args.input_width, :]
            # depth_gt = depth_gt[-self.args.input_height:, lm: lm + self.args.input_width, :]

            # augment image
            image_aug = image.copy()
            image_aug = self.random_color_augment(image_aug)
            image_aug = np.clip(image_aug, 0, 255).astype(np.float32) / 255.0
            image = image.astype(np.float32) / 255.0

            # recover unit
            depth_gt = depth_gt / 256.0

            if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                raise ValueError('Size mismatches.')
                # image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
            image, image_aug, depth_gt = self.train_preprocess(image, image_aug, depth_gt)

            # horizontal translation
            image, image_aug, depth_gt = self.translateX(image, image_aug, depth_gt)

            # # https://github.com/ShuweiShao/URCDC-Depth
            image, image_aug, depth_gt = self.Cut_Flip(image, image_aug, depth_gt)
            sample = {'image': image, 'image_aug': image_aug, 'depth': depth_gt, 'focal': focal}
        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path_test

            image_path = os.path.join(data_path, sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                depth_path = image_path.replace('/image/', '/groundtruth_depth/').replace('sync_image', 'sync_groundtruth_depth')
                depth_gt = Image.open(depth_path)

                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                depth_gt = depth_gt / 256.0

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                # crop image again
                lm = (1216 - self.args.input_width) // 2
                image = image[-self.args.input_height:, lm: lm + self.args.input_width, :]

                if self.mode == 'online_eval':
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': True}
            else:
                sample = {'image': image, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)

        return sample

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
        max_tx = self.args.max_translation_x
        candidates = list(range(-(max_tx - 1), max_tx))
        tx = random.choice(candidates)
        return translate(img, tx), translate(img_aug, tx), translate(gt, tx)

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

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

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def Cut_Flip(self, image, image_aug, depth):

        p = random.random()
        if p < 0.5:
            return image, image_aug, depth
        image_copy = copy.deepcopy(image)
        image_aug_copy = copy.deepcopy(image_aug)
        depth_copy = copy.deepcopy(depth)
        h, w, c = image.shape

        N = 2
        h_list = []
        h_interval_list = []  # hight interval
        for i in range(N - 1):
            h_list.append(random.randint(int(0.2 * h), int(0.8 * h)))
        h_list.append(h)
        h_list.append(0)
        h_list.sort()
        h_list_inv = np.array([h] * (N + 1)) - np.array(h_list)
        for i in range(len(h_list) - 1):
            h_interval_list.append(h_list[i + 1] - h_list[i])
        for i in range(N):
            image[h_list[i]:h_list[i + 1], :, :] = image_copy[h_list_inv[i] - h_interval_list[i]:h_list_inv[i], :, :]
            image_aug[h_list[i]:h_list[i + 1], :, :] = image_aug_copy[h_list_inv[i] - h_interval_list[i]:h_list_inv[i], :, :]
            depth[h_list[i]:h_list[i + 1], :, :] = depth_copy[h_list_inv[i] - h_interval_list[i]:h_list_inv[i], :, :]

        return image, image_aug, depth

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            image_aug = self.to_tensor(sample['image_aug'])
            image_aug = self.normalize(image_aug)

            depth = self.to_tensor(depth)
            return {'image': image, 'image_aug': image_aug, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
