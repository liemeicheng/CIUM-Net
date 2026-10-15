"""Paired RGB metrics and unpaired NIQE/BRISQUE evaluation."""
import argparse
import math
import numpy as np
from dataloader import pair_paths, image_paths, read_rgb
from metrics_core import PairedMetrics, calculate_niqe
from utils import resolve_device, save_json


def paired_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True)
    p.add_argument('--targets', required=True)
    p.add_argument('--pairing', choices=['same', 'lolv2-real'], default='same')
    p.add_argument('--device', default='auto')
    p.add_argument('--skip-lpips', action='store_true', help='Partial evaluation only')
    p.add_argument('--output', default='results/paired_metrics.json')
    args = p.parse_args()
    pairs = pair_paths(args.predictions, args.targets, args.pairing)
    metric = PairedMetrics(resolve_device(args.device), not args.skip_lpips)
    rows = [dict(image=a.name, **metric(read_rgb(a), read_rgb(b))) for a, b in pairs]
    mean = {key: sum(row[key] for row in rows)/len(rows) for key in rows[0] if key != 'image'}
    save_json(args.output, dict(protocol='RGB, no GT-mean correction, no resize, no border crop',
                               complete=not args.skip_lpips, count=len(rows), mean=mean, images=rows))
    print(mean)


def unpaired_main():
    p = argparse.ArgumentParser(description='NIQE/BRISQUE with equal dataset averaging')
    p.add_argument('--dataset', action='append', required=True, metavar='NAME=FOLDER',
                   help='Repeat for DICM, LIME, MEF, NPE and VV predictions')
    p.add_argument('--niqe-only', action='store_true', help='Partial evaluation without BRISQUE')
    p.add_argument('--output', default='results/unpaired_metrics.json')
    args = p.parse_args()
    if not args.niqe_only:
        try:
            from imquality import brisque as _dependency_check
            from metrics_core import brisque_score
        except ImportError as exc:
            raise ImportError('Install requirements.txt to compute BRISQUE') from exc
    datasets = {}
    for entry in args.dataset:
        name, sep, folder = entry.partition('=')
        if not sep or not name or name in datasets:
            p.error('Use unique NAME=FOLDER entries')
        rows = []
        for path in image_paths(folder):
            rgb = read_rgb(path)
            if (rgb.shape[0]//96)*(rgb.shape[1]//96) < 2:
                raise ValueError(f'{path}: NIQE needs at least two complete 96x96 blocks')
            # The inherited NIQE implementation takes BGR, while PIL loads RGB.
            row = dict(image=path.name, niqe=calculate_niqe(rgb[..., ::-1].copy()))
            if not args.niqe_only:
                row['brisque'] = brisque_score(rgb)
            if any(not math.isfinite(v) for k, v in row.items() if k != 'image'):
                raise ValueError(f'Non-finite metric for {path}; no images were silently dropped')
            rows.append(row)
        means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'image'}
        datasets[name] = dict(count=len(rows), mean=means, images=rows)
    keys = next(iter(datasets.values()))['mean']
    avg = {k: float(np.mean([d['mean'][k] for d in datasets.values()])) for k in keys}
    save_json(args.output, dict(complete=not args.niqe_only, datasets=datasets,
                               equal_weight_dataset_average=avg))
    print(avg)


if __name__ == '__main__':
    import sys
    if any(arg == '--dataset' or arg.startswith('--dataset=') for arg in sys.argv[1:]):
        unpaired_main()
    else:
        paired_main()
