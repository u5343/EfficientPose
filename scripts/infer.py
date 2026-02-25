import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from efficientpose.model import PoseRegressionNet
from efficientpose.utils import Visualizer, compute_rotation_matrix_from_ortho6d


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-image pose inference.")
    parser.add_argument("--weights", required=True, help="Path to checkpoint file")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="inference_result.jpg", help="Output image path")
    parser.add_argument("--backbone", default="efficientnet_b2")
    parser.add_argument(
        "--cam-k",
        nargs=9,
        type=float,
        default=[
            913.00958,
            0.0,
            649.93371,
            0.0,
            912.7648,
            369.83358,
            0.0,
            0.0,
            1.0,
        ],
        help="Camera intrinsics as 9 values row-major",
    )
    parser.add_argument("--cam-k-info-yml", default="", help="Optional info.yml path")
    parser.add_argument("--cam-k-frame-id", type=int, default=0, help="Frame id for info.yml cam_K")
    parser.add_argument("--offset", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--axis-length", type=float, default=0.1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(arg_device):
    if arg_device == "cpu":
        return torch.device("cpu")
    if arg_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cam_k(args):
    if args.cam_k_info_yml:
        with open(args.cam_k_info_yml, "r", encoding="utf-8") as f:
            info_data = yaml.load(f, Loader=yaml.FullLoader)
        return np.array(info_data[args.cam_k_frame_id]["cam_K"], dtype=np.float32).reshape(3, 3)

    return np.array(args.cam_k, dtype=np.float32).reshape(3, 3)


def load_model(weights_path, device, backbone):
    model = PoseRegressionNet(backbone_name=backbone, pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_raw = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    return img_raw, tensor.unsqueeze(0)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    model = load_model(args.weights, device, args.backbone)
    cam_k = load_cam_k(args)

    img_raw, img_tensor = preprocess_image(args.image)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        pred_r_6d, pred_t = model(img_tensor)
        pred_r = compute_rotation_matrix_from_ortho6d(pred_r_6d)[0].cpu().numpy()
        pred_t = pred_t[0].cpu().numpy()

    vis = Visualizer()
    output = vis.draw_shifted_axis(
        img_raw.copy(),
        pred_r,
        pred_t,
        cam_k,
        np.array(args.offset, dtype=np.float32),
        axis_length=args.axis_length,
    )

    cv2.imwrite(args.output, output)
    print(f"Predicted translation (m): {pred_t}")
    print(f"Saved visualization: {args.output}")


if __name__ == "__main__":
    main()
