import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from efficientpose.dataset import LinemodDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset smoke test")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--list-file",
        default="",
        help="Optional id list file; if omitted and <data-root>/train.txt is missing, use all rgb/*.png",
    )
    parser.add_argument("--samples", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    list_file = Path(args.list_file) if args.list_file else None
    default_list = data_root / "train.txt"

    if list_file is None and default_list.exists():
        list_file = default_list
    if list_file and not list_file.exists():
        raise FileNotFoundError(f"List file not found: {list_file}")

    ds = LinemodDataset(data_root, list_file=list_file)
    print(f"Dataset size: {len(ds)}")

    n = min(args.samples, len(ds))
    for i in range(n):
        sample = ds[i]
        print(
            f"{i}: image={tuple(sample['image'].shape)}, "
            f"R={tuple(sample['gt_R'].shape)}, "
            f"t={sample['gt_t'].tolist()}"
        )

    print("Dataset smoke test passed")


if __name__ == "__main__":
    main()
