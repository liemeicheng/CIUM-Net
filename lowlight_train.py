"""Train CIUM-Net with the paper settings, or resume a complete checkpoint."""
import argparse
import csv
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import PairedImages
from lowlight_test import enhance
from Myloss import RestorationLoss
from metrics_core import PairedMetrics
from model import CIUMNet
from loss_weights import paper_config
from utils import learning_rate, load_model, resolve_device, save_json, seed_everything, seed_worker


@torch.no_grad()
def validate(model, loader, metric, device):
    model.eval()
    totals = {}
    count = 0
    for low, target, _ in loader:
        prediction = enhance(model, low.to(device))
        reference = target[0].permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
        for key, value in metric(prediction, reference).items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value/count for key, value in totals.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['lolv1', 'lolv2-real', 'lolv2-synthetic'], default='lolv1')
    p.add_argument('--train-low', required=True)
    p.add_argument('--train-normal', required=True)
    p.add_argument('--val-low')
    p.add_argument('--val-normal')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume')
    p.add_argument('--skip-lpips', action='store_true')
    p.add_argument('--no-lcum', action='store_true', help='Table 7: disable LCUM')
    p.add_argument('--no-iecc', action='store_true', help='Table 7: disable IECC')
    p.add_argument('--iecc-variant', choices=['E0', 'E1', 'E2', 'E3', 'E4'],
                   help='Table 9: also disables LCUM; E0 retains chromatic L1 supervision')
    args = p.parse_args()
    if bool(args.val_low) != bool(args.val_normal):
        p.error('Provide both validation directories, or neither')
    if args.no_iecc and args.iecc_variant:
        p.error('--no-iecc and --iecc-variant are mutually exclusive')
    device = resolve_device(args.device)
    config = paper_config(args.dataset)
    config.setdefault('model', {})
    config.setdefault('loss', {})
    if args.no_lcum or args.iecc_variant:
        config['model']['use_lcum'] = False
    if args.no_iecc:
        config['loss']['iecc'] = 0.0
    if args.iecc_variant:
        variants = {'E0': (0, 0, 0), 'E1': (1, 0, 0), 'E2': (1, 0.5, 0),
                    'E3': (1, 0, 0.01), 'E4': (1, 0.5, 0.01)}
        config['loss'].update(zip(('lambda_mu', 'lambda_var', 'lambda_cov'), variants[args.iecc_variant]))
        config['loss']['iecc'] = 0.005
    cfg = config['training']
    if cfg['epochs'] < 1 or cfg['batch_size'] < 1 or cfg['validate_every'] < 1:
        raise ValueError('epochs, batch_size and validate_every must be positive')
    if not 0 <= cfg['warmup_epochs'] < cfg['epochs']:
        raise ValueError('warmup_epochs must be between zero and epochs-1')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Output directory is not empty; choose a new run directory or use --resume')
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_set = PairedImages(args.train_low, args.train_normal, config['pairing'], cfg['patch_size'])
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=args.workers, worker_init_fn=seed_worker,
                              generator=generator, pin_memory=device.type == 'cuda')
    val_loader = None
    if args.val_low:
        val_set = PairedImages(args.val_low, args.val_normal, config['pairing'])
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.workers)
    metric = PairedMetrics(device, not args.skip_lpips) if val_loader is not None else None
    model = CIUMNet(**config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0)
    criterion = RestorationLoss(**config['loss']).to(device)
    start, best = 1, {}
    if args.resume:
        resumed, checkpoint = load_model(args.resume, device)
        if checkpoint['config'] != config:
            raise ValueError('Resume config differs from the saved config')
        model.load_state_dict(resumed.state_dict())
        del resumed
        optimizer.load_state_dict(checkpoint['optimizer'])
        start, best = checkpoint['epoch']+1, checkpoint['best']
        random.setstate(checkpoint['rng']['python'])
        np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        generator.set_state(checkpoint['rng']['loader'])
        if device.type == 'cuda' and checkpoint['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
    if start > cfg['epochs']:
        raise ValueError('Checkpoint has already completed the configured number of epochs')
    save_json(output/'config.json', config)
    save_json(output/'run.json', dict(arguments=vars(args), torch=torch.__version__,
                                     device=str(device), train_pairs=len(train_set)))
    history = output/'history.csv'
    with history.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'lr', 'loss', 'psnr', 'ssim', 'lpips'])
        if file.tell() == 0:
            writer.writeheader()
        for epoch in range(start, cfg['epochs']+1):
            lr = learning_rate(epoch, cfg['epochs'], cfg['warmup_epochs'], cfg['lr'], cfg['min_lr'])
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            loss_sum, count = 0.0, 0
            for low, target, _ in train_loader:
                low, target = low.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred_rgb, pred_hvi = model(low, return_hvi=True)
                loss = criterion(pred_rgb, pred_hvi, target, model.HVIT(target))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError(f'Non-finite gradient at epoch {epoch}')
                optimizer.step()
                loss_sum += loss.item()*len(low)
                count += len(low)
            scores = validate(model, val_loader, metric, device) if val_loader is not None and (
                epoch % cfg['validate_every'] == 0 or epoch == cfg['epochs']) else {}
            improved = []
            for key, value in scores.items():
                if key not in best or (value < best[key]['value'] if key == 'lpips' else value > best[key]['value']):
                    best[key] = dict(value=value, epoch=epoch)
                    improved.append(key)
            checkpoint = dict(format='cium-net-paper-v1', model_config=model.model_config,
                              model=model.state_dict(), optimizer=optimizer.state_dict(),
                              config=config, epoch=epoch, best=best,
                              rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                       torch=torch.get_rng_state(), loader=generator.get_state(),
                                       cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None))
            temp = output/'last.tmp'
            torch.save(checkpoint, temp)
            temp.replace(output/'last.pt')
            for key in improved:
                torch.save(checkpoint, output/f'best_{key}.pt')
            row = dict(epoch=epoch, lr=lr, loss=loss_sum/count, **scores)
            writer.writerow(row)
            file.flush()
            save_json(output/'best.json', best)
            print(row, flush=True)


if __name__ == '__main__':
    main()
