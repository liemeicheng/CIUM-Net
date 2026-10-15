"""Report CIUM-Net parameter counts without training or a forward pass."""
import argparse
from model import CIUMNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-lcum', action='store_true')
    args = parser.parse_args()
    model = CIUMNet(use_lcum=not args.no_lcum)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f'Total parameters: {total:,} ({total/1e6:.4f} M)')
    print(f'Trainable parameters: {trainable:,}')
    print('LCUM parameters: 0 (input-derived, parameter-free modulation)')


if __name__ == '__main__':
    main()
