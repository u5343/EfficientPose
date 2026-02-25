import argparse
from pathlib import Path

import numpy as np
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate model center and OBB orientation")
    parser.add_argument("--model-path", required=True, help="Path to .ply mesh")
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    raw_center = np.mean(mesh.vertices, axis=0)

    try:
        obb = mesh.bounding_box_oriented
        transform = obb.primitive.transform
        raw_rotation = transform[:3, :3]

        extents = obb.primitive.extents
        sort_indices = np.argsort(extents)
        axis_z = raw_rotation[:, sort_indices[2]]
        axis_x = raw_rotation[:, sort_indices[1]]
        axis_y = np.cross(axis_z, axis_x)
        raw_rotation = np.column_stack((axis_x, axis_y, axis_z))
    except Exception:
        raw_rotation = np.eye(3)

    print("RAW_CENTER = np.array([")
    print(f"    {raw_center[0]:.5f}, {raw_center[1]:.5f}, {raw_center[2]:.5f}")
    print("], dtype=np.float32)\n")

    print("RAW_ROTATION = np.array([")
    print(f"    [{raw_rotation[0,0]:.5f}, {raw_rotation[0,1]:.5f}, {raw_rotation[0,2]:.5f}],")
    print(f"    [{raw_rotation[1,0]:.5f}, {raw_rotation[1,1]:.5f}, {raw_rotation[1,2]:.5f}],")
    print(f"    [{raw_rotation[2,0]:.5f}, {raw_rotation[2,1]:.5f}, {raw_rotation[2,2]:.5f}]")
    print("], dtype=np.float32)")


if __name__ == "__main__":
    main()
