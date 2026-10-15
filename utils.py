"""Small shared runtime helpers. No downloads or GPU selection at import time."""
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from model import CIUMNet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(name):
    device = torch.device(name if name != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; use --device cpu for a smoke test')
    return device


def learning_rate(epoch, epochs=1000, warmup_epochs=3, lr=1e-4, min_lr=1e-7):
    """1-based epochs: linear warm-up, then cosine decay ending at min_lr."""
    if epoch <= warmup_epochs:
        return lr * epoch / warmup_epochs
    progress = (epoch-warmup_epochs) / max(1, epochs-warmup_epochs)
    return min_lr + 0.5*(lr-min_lr)*(1+math.cos(math.pi*progress))


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get('format') != 'cium-net-paper-v1':
        raise ValueError('Expected a CIUM-Net paper-v1 checkpoint produced by lowlight_train.py. '
                         'Legacy CIDNet/DIEM checkpoints are not compatible with the corrected graph.')
    model = CIUMNet(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device), checkpoint


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # JSON has no infinity/NaN literals: use descriptive strings for perfect PSNR.
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return str(x)
        return x
    path.write_text(json.dumps(clean(data), indent=2, ensure_ascii=False), encoding='utf-8')
