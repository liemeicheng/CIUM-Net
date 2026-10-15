"""Paper loss weights and shared training defaults (Sections 3.3 and 4.1.2)."""
from copy import deepcopy

LOSS_WEIGHTS = dict(l1=1.0, ssim=0.5, edge=50.0, hvi=1.0, iecc=0.005,
                    lambda_mu=1.0, lambda_var=0.5, lambda_cov=0.01)
MODEL_CONFIG = dict(channels=[36, 36, 72, 144], heads=[1, 2, 4, 8], use_lcum=True)
TRAINING_CONFIG = dict(epochs=1000, batch_size=8, patch_size=256, lr=1e-4,
                       min_lr=1e-7, warmup_epochs=3, validate_every=10)


def paper_config(dataset='lolv1'):
    if dataset not in ('lolv1', 'lolv2-real', 'lolv2-synthetic'):
        raise ValueError(f'Unsupported paired dataset: {dataset}')
    return dict(dataset=dataset, pairing='lolv2-real' if dataset == 'lolv2-real' else 'same',
                model=deepcopy(MODEL_CONFIG), loss=deepcopy(LOSS_WEIGHTS),
                training=deepcopy(TRAINING_CONFIG))
