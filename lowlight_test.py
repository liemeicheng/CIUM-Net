"""Enhance a folder using a trained CIUM-Net checkpoint."""
import argparse
import os
from pathlib import Path
import torch
from PIL import Image
from dataloader import image_paths, read_rgb, to_tensor
from utils import load_model, resolve_device


@torch.no_grad()
def enhance(model, tensor):
    model.eval()
    output = model(tensor).clamp(0, 1)
    return output[0].permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    source, output = Path(os.path.abspath(args.input)), Path(os.path.abspath(args.output))
    if source == output or (output.exists() and os.path.samefile(source, output)):
        raise ValueError('Output directory must differ from input directory')
    paths = image_paths(source)
    device = resolve_device(args.device)
    model, _ = load_model(args.checkpoint, device)
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        result = enhance(model, to_tensor(read_rgb(path))[None].to(device))
        Image.fromarray(result).save(output / (path.stem + '.png'))
    print(f'Saved {len(paths)} lossless PNG images to {output}')


if __name__ == '__main__':
    main()
