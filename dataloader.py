"""Explicit file pairing and synchronized crops/flips for the LOL datasets."""
from pathlib import Path
import random
import re
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def image_paths(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Image directory not found: {folder}')
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f'No supported images in {folder}')
    stems = [p.stem.casefold() for p in paths]
    if len(set(stems)) != len(stems):
        raise ValueError(f'Duplicate image stems in {folder}; output names would collide')
    return paths


def pair_paths(low_dir, normal_dir, naming='same'):
    low, normal = image_paths(low_dir), image_paths(normal_dir)
    def key(path, reference=False):
        stem = path.stem.casefold()
        if naming == 'lolv2-real':
            stem = re.sub(r'^(normal|low)', '', stem)
        elif naming != 'same':
            raise ValueError(f'Unknown pairing convention: {naming}')
        return stem
    refs = {key(p, True): p for p in normal}
    keys = [key(p) for p in low]
    if len(refs) != len(normal) or len(set(keys)) != len(keys):
        raise ValueError('Ambiguous normalized image names')
    if set(keys) != set(refs):
        raise ValueError(f'Unmatched pairs: missing targets={sorted(set(keys)-set(refs))[:5]}, '
                         f'extra targets={sorted(set(refs)-set(keys))[:5]}')
    return [(p, refs[key(p)]) for p in low]


def read_rgb(path):
    with Path(path).open('rb') as stream:
        with Image.open(stream) as image:
            return np.array(image.convert('RGB'), dtype=np.uint8)


def to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255


class PairedImages(Dataset):
    def __init__(self, low_dir, normal_dir, naming='same', patch_size=None):
        self.pairs = pair_paths(low_dir, normal_dir, naming)
        self.patch_size = patch_size
        if patch_size is not None and (patch_size < 8 or patch_size % 8):
            raise ValueError('Training patch size must be a positive multiple of 8')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        low_path, normal_path = self.pairs[index]
        low, normal = read_rgb(low_path), read_rgb(normal_path)
        if low.shape != normal.shape:
            raise ValueError(f'Pair shape mismatch: {low_path}, {normal_path}')
        if self.patch_size:
            size = self.patch_size
            h, w = low.shape[:2]
            if min(h, w) < size:
                raise ValueError(f'{low_path} is smaller than patch size {size}')
            top, left = random.randint(0, h-size), random.randint(0, w-size)
            low, normal = low[top:top+size, left:left+size], normal[top:top+size, left:left+size]
            if random.random() < 0.5:
                low, normal = low[:, ::-1], normal[:, ::-1]
            if random.random() < 0.5:
                low, normal = low[::-1], normal[::-1]
        return to_tensor(low), to_tensor(normal), low_path.name
