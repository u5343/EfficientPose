# EfficientPoseW

轻量级 6D 姿态回归项目（基于 EfficientNet + 6D rotation）。

本项目已移除所有 ArUco 相关流程，训练集制作与训练不再依赖 ArUco。

## 目录结构

- `efficientpose/`: 核心库代码（数据集、模型、损失、工具）
- `scripts/`: 可执行脚本（训练、推理、数据集冒烟测试）
- `tools/`: 独立工具脚本
- `pretrained/`: 本地主干网络预训练权重
- `runs/`: 训练输出目录

兼容入口（根目录）:
- `train.py` -> `scripts/train.py`
- `infer.py` / `inference!!.py` -> `scripts/infer.py`
- `testdataset.py` -> `scripts/test_dataset.py`
- `calculate_offset.py` -> `tools/calculate_offset.py`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练

```bash
python scripts/train.py --data-root D:\Linemod_BlenderNew\Linemod_preprocessed\lm\train_pbr\000015 --epochs 300 --batch-size 4
```

常用参数:
- `--train-list` / `--val-list`: 自定义 train/test txt（必须成对提供）
- 若未提供且 `<data-root>/train.txt`、`test.txt` 不存在：自动从 `rgb/*.png` 按 `--val-ratio` 切分
- `--resume`: 从 `last.pth` 或权重文件续训
- `--output-root`: 输出目录（默认 `runs/train`）
- `--no-pretrained`: 不加载主干预训练

损失函数说明（对称体模式）:
- 平移损失：约束原点坐标
- 姿态损失：仅约束预测旋转矩阵的 Y 轴方向与 GT Y 轴方向一致
- 不再对完整 3x3 旋转矩阵逐元素监督

## 推理

```bash
python scripts/infer.py --weights D:\EfficientPoseW\runs\train\exp47\weights\best.pth --image D:\EfficientPoseW\test-pngs\0000.png --output inference_result.jpg
```

可选参数:
- `--cam-k` 9 个数字（行优先）
- `--cam-k-info-yml` + `--cam-k-frame-id` 从 `info.yml` 读取内参
- `--offset` 可视化偏移（米）

## 数据集检查

```bash
python3 scripts/test_dataset.py --data-root /path/to/LINEMOD/object_dir
```

## 偏移辅助工具

```bash
python3 tools/calculate_offset.py --model-path /path/to/model.ply
```
