import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]

    x = x_raw / (torch.norm(x_raw, dim=1, keepdim=True) + 1e-8)
    z = torch.cross(x, y_raw, dim=1)
    z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-8)
    y = torch.cross(z, x, dim=1)

    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    z = z.unsqueeze(2)
    return torch.cat((x, y, z), 2)


class Visualizer:
    def denormalize(self, tensor_img):
        img = tensor_img.permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _project(points_local, r_mat, t_vec, cam_k):
        points_cam = (r_mat @ points_local.T).T + t_vec
        points_2d_h = (cam_k @ points_cam.T).T
        points_2d = points_2d_h[:, :2] / (points_2d_h[:, 2:3] + 1e-6)
        return points_2d.astype(int)

    def draw_axis(self, img, r_mat, t_vec, cam_k, axis_length=0.1):
        if torch.is_tensor(r_mat):
            r_mat = r_mat.detach().cpu().numpy()
        if torch.is_tensor(t_vec):
            t_vec = t_vec.detach().cpu().numpy()
        if torch.is_tensor(cam_k):
            cam_k = cam_k.detach().cpu().numpy()

        points_local = np.float32(
            [
                [0, 0, 0],
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, axis_length],
            ]
        )
        points_2d = self._project(points_local, r_mat, t_vec, cam_k)
        origin = tuple(points_2d[0])

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i in range(1, 4):
            cv2.line(img, origin, tuple(points_2d[i]), colors[i - 1], 3)

        cv2.circle(img, origin, 5, (255, 255, 255), -1)
        return img

    def draw_axis_gt(self, img, r_mat, t_vec, cam_k, axis_length=0.1):
        if torch.is_tensor(r_mat):
            r_mat = r_mat.detach().cpu().numpy()
        if torch.is_tensor(t_vec):
            t_vec = t_vec.detach().cpu().numpy()
        if torch.is_tensor(cam_k):
            cam_k = cam_k.detach().cpu().numpy()

        points_local = np.float32(
            [
                [0, 0, 0],
                [axis_length, 0, 0],
                [0, axis_length, 0],
                [0, 0, axis_length],
            ]
        )
        points_2d = self._project(points_local, r_mat, t_vec, cam_k)
        origin = tuple(points_2d[0])

        for i in range(1, 4):
            cv2.line(img, origin, tuple(points_2d[i]), (0, 255, 0), 5)

        cv2.circle(img, origin, 7, (0, 255, 0), -1)
        return img

    def draw_shifted_axis(self, img, r_mat, t_vec, cam_k, offset, axis_length=0.1):
        if torch.is_tensor(r_mat):
            r_mat = r_mat.detach().cpu().numpy()
        if torch.is_tensor(t_vec):
            t_vec = t_vec.detach().cpu().numpy()
        if torch.is_tensor(cam_k):
            cam_k = cam_k.detach().cpu().numpy()

        origin_local = np.asarray(offset, dtype=np.float32)
        points_local = np.float32(
            [
                origin_local,
                origin_local + np.array([axis_length, 0, 0], dtype=np.float32),
                origin_local + np.array([0, axis_length, 0], dtype=np.float32),
                origin_local + np.array([0, 0, axis_length], dtype=np.float32),
            ]
        )
        points_2d = self._project(points_local, r_mat, t_vec, cam_k)
        origin = tuple(points_2d[0])

        cv2.line(img, origin, tuple(points_2d[1]), (0, 0, 255), 3)
        cv2.line(img, origin, tuple(points_2d[2]), (0, 255, 0), 3)
        cv2.line(img, origin, tuple(points_2d[3]), (255, 0, 0), 3)
        cv2.circle(img, origin, 5, (255, 255, 255), -1)
        return img


def plot_training_results(train_loss, val_loss, t_error, save_path="results.png", y_error=None):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, t_error, label="T Error (cm)", color="orange")
    if y_error is not None and len(y_error) == len(t_error):
        plt.plot(epochs, y_error, label="Y Axis Error (deg)", color="green")
    plt.title("Pose Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
