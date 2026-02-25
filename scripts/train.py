import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from efficientpose.dataset import LinemodDataset
from efficientpose.loss import PoseLoss
from efficientpose.model import PoseRegressionNet
from efficientpose.utils import (
    Visualizer,
    compute_rotation_matrix_from_ortho6d,
    plot_training_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train pose regression model.")
    parser.add_argument("--data-root", required=True, help="Path to LINEMOD object data folder")
    parser.add_argument("--train-list", default=None, help="Path to train id list txt")
    parser.add_argument("--val-list", default=None, help="Path to val/test id list txt")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio used only when list files are missing",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for auto split when list files are missing",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone", default="efficientnet_b2")
    parser.add_argument(
        "--local-pretrained",
        default="pretrained/efficientnet_b2_ra-bcdf34b7.pth",
        help="Local timm backbone weight path",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Disable backbone pretraining")
    parser.add_argument("--resume", default="", help="Resume checkpoint path")
    parser.add_argument("--output-root", default="runs/train", help="Root output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--vis-max", type=int, default=4, help="Max validation images in vis grid")
    return parser.parse_args()


def resolve_device(arg_device):
    if arg_device == "cpu":
        return torch.device("cpu")
    if arg_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_lists(data_root, train_list, val_list):
    data_root = Path(data_root)
    train_list = Path(train_list) if train_list else None
    val_list = Path(val_list) if val_list else None
    return data_root, train_list, val_list


def auto_split_ids(data_root, val_ratio, split_seed):
    rgb_dir = data_root / "rgb"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"rgb folder not found: {rgb_dir}")

    ids = sorted(p.stem for p in rgb_dir.glob("*.png"))
    if len(ids) < 2:
        raise ValueError("Need at least 2 images for automatic train/val split")

    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(len(ids), generator=generator).tolist()

    val_count = max(1, int(round(len(ids) * val_ratio)))
    val_count = min(val_count, len(ids) - 1)

    val_idx = set(indices[:val_count])
    train_ids = [ids[i] for i in range(len(ids)) if i not in val_idx]
    val_ids = [ids[i] for i in range(len(ids)) if i in val_idx]
    return train_ids, val_ids


def get_save_dir(output_root):
    base = Path(output_root)
    base.mkdir(parents=True, exist_ok=True)
    exps = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("exp"):
            suffix = p.name.replace("exp", "")
            if suffix.isdigit():
                exps.append(int(suffix))
    next_id = max(exps) + 1 if exps else 1
    return base / f"exp{next_id}"


def load_resume_if_needed(model, criterion, optimizer, resume_path, device):
    start_epoch = 0
    best_t_error = float("inf")
    history = {"train_loss": [], "val_loss": [], "t_error": [], "y_err_deg": []}

    if not resume_path:
        return start_epoch, best_t_error, history

    if not os.path.isfile(resume_path):
        print(f"Resume file not found: {resume_path}, training from scratch")
        return start_epoch, best_t_error, history

    checkpoint = torch.load(resume_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError:
                print(
                    "Optimizer state in checkpoint is incompatible with current setup, "
                    "optimizer will be re-initialized."
                )
        if "loss_state_dict" in checkpoint:
            criterion.load_state_dict(checkpoint["loss_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        history = checkpoint.get("history", history)
        best_t_error = checkpoint.get("best_t_error", best_t_error)
        print(f"Resumed full checkpoint: {resume_path}, start epoch {start_epoch}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Resumed weight-only checkpoint: {resume_path}")

    return start_epoch, best_t_error, history


def main():
    args = parse_args()
    data_root, train_list, val_list = resolve_lists(args.data_root, args.train_list, args.val_list)

    save_dir = get_save_dir(args.output_root)
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    use_amp = device.type == "cuda"

    print(f"Output dir: {save_dir}")
    print(f"Device: {device}")

    default_train = data_root / "train.txt"
    default_val = data_root / "test.txt"
    has_default_lists = default_train.exists() and default_val.exists()

    train_ids = None
    val_ids = None
    if (train_list is None) ^ (val_list is None):
        raise ValueError("Please provide both --train-list and --val-list, or neither")

    if train_list and val_list:
        if not train_list.exists():
            raise FileNotFoundError(f"train list not found: {train_list}")
        if not val_list.exists():
            raise FileNotFoundError(f"val list not found: {val_list}")
        print(f"Using explicit list files: {train_list}, {val_list}")
    elif has_default_lists:
        train_list = default_train
        val_list = default_val
        print(f"Using default list files: {train_list}, {val_list}")
    else:
        train_ids, val_ids = auto_split_ids(data_root, args.val_ratio, args.split_seed)
        print(
            "List files not found, using auto split from rgb/: "
            f"train={len(train_ids)}, val={len(val_ids)}"
        )

    train_ds = LinemodDataset(data_root, list_file=train_list, ids=train_ids)
    val_ds = LinemodDataset(data_root, list_file=val_list, ids=val_ids)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    model = PoseRegressionNet(
        backbone_name=args.backbone,
        pretrained=not args.no_pretrained,
        local_weight_path=args.local_pretrained,
    ).to(device)
    criterion = PoseLoss().to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
    )
    scaler = GradScaler(enabled=use_amp)
    visualizer = Visualizer()

    start_epoch, best_t_error, history = load_resume_if_needed(
        model, criterion, optimizer, args.resume, device
    )
    history.setdefault("y_err_deg", [])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_sum = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in loop:
            imgs = batch["image"].to(device)
            gt_r = batch["gt_R"].to(device)
            gt_t = batch["gt_t"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp, dtype=torch.float16):
                pred_r, pred_t = model(imgs)
                loss, _, _ = criterion(pred_r, pred_t, gt_r, gt_t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_loss_sum = 0.0
        val_t_err_sum = 0.0
        val_y_err_sum = 0.0
        do_vis = True

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                gt_r = batch["gt_R"].to(device)
                gt_t = batch["gt_t"].to(device)
                cam_k = batch["cam_K"].to(device)

                pred_r, pred_t = model(imgs)
                loss, _, _ = criterion(pred_r, pred_t, gt_r, gt_t)
                val_loss_sum += loss.item()

                t_err_cm = torch.norm(pred_t - gt_t, dim=1).mean().item() * 100.0
                val_t_err_sum += t_err_cm

                pred_r_mat = compute_rotation_matrix_from_ortho6d(pred_r)
                pred_y = pred_r_mat[:, :, 1]
                gt_y = gt_r[:, :, 1]
                cos_sim = torch.nn.functional.cosine_similarity(pred_y, gt_y, dim=1)
                cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                y_err_deg = torch.rad2deg(torch.acos(cos_sim)).mean().item()
                val_y_err_sum += y_err_deg

                if do_vis:
                    vis_list = []
                    for i in range(min(args.vis_max, len(imgs))):
                        img_vis = visualizer.denormalize(imgs[i])
                        img_vis = visualizer.draw_axis_gt(img_vis.copy(), gt_r[i], gt_t[i], cam_k[i])
                        img_vis = visualizer.draw_axis(img_vis, pred_r_mat[i], pred_t[i], cam_k[i])
                        err = torch.norm(pred_t[i] - gt_t[i]).item() * 100.0
                        cv2.putText(
                            img_vis,
                            f"Err:{err:.1f}cm",
                            (5, 20),
                            0,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                        vis_list.append(img_vis)

                    if vis_list:
                        grid = np.hstack(vis_list)
                        cv2.imwrite(str(save_dir / f"val_epoch_{epoch}.jpg"), grid)
                    do_vis = False

        avg_train_loss = train_loss_sum / max(len(train_loader), 1)
        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        avg_t_error = val_t_err_sum / max(len(val_loader), 1)
        avg_y_error = val_y_err_sum / max(len(val_loader), 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["t_error"].append(avg_t_error)
        history["y_err_deg"].append(avg_y_error)

        print(
            f"Validation: loss={avg_val_loss:.4f}, "
            f"t_err={avg_t_error:.2f}cm, y_err={avg_y_error:.2f}deg"
        )

        plot_training_results(
            history["train_loss"],
            history["val_loss"],
            history["t_error"],
            save_dir / "results.png",
            y_error=history["y_err_deg"],
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_state_dict": criterion.state_dict(),
            "history": history,
            "best_t_error": min(best_t_error, avg_t_error),
        }
        torch.save(checkpoint, weights_dir / "last.pth")

        if avg_t_error < best_t_error:
            best_t_error = avg_t_error
            torch.save(model.state_dict(), weights_dir / "best.pth")
            print("Best model updated")

    print(f"Training complete. Best translation error: {best_t_error:.2f}cm")


if __name__ == "__main__":
    main()
