# AGENT.md

## 项目目标
本项目用于 LINEMOD 风格数据上的 6D 姿态回归训练与推理：
- 输入 RGB 图像
- 预测旋转（6D 表示）+ 平移（3D）
- 支持训练可视化与单图推理可视化

注意：本项目已完全移除 ArUco 相关代码与流程。

## 目录职责
- `efficientpose/`: 核心库代码
  - `dataset.py`: `LinemodDataset`
  - `model.py`: `PoseRegressionNet`
  - `loss.py`: `PoseLoss`
  - `utils.py`: 旋转变换/绘图工具
- `scripts/`: CLI 脚本
  - `train.py`: 训练入口（支持断点续训）
  - `infer.py`: 推理入口（支持完整 checkpoint 与纯权重）
  - `test_dataset.py`: 数据集冒烟测试
- `tools/`
  - `calculate_offset.py`: mesh 中心/OBB 辅助工具
- `pretrained/`: 本地 backbone 权重
- `runs/`: 训练产物输出目录

兼容入口（根目录脚本）保留为包装器，转发到上述新路径。

## 数据约定
`LinemodDataset` 默认读取：
- `<data_root>/rgb/<id>.png`
- `<data_root>/train.txt`、`<data_root>/test.txt`（可选）
- 若列表文件不存在，可直接从 `rgb/*.png` 生成样本并在训练脚本中自动划分 train/val
- `<data_root>/gt.yml` 与 `<data_root>/info.yml`
  - 若不存在，自动尝试 `<data_root>/..`

关键单位约束：
- `cam_t_m2c` 从毫米转换为米（`/1000.0`）
- 训练/验证/推理阶段都必须保持米制一致

## Checkpoint 约定
- `last.pth`: 完整训练状态（模型、优化器、loss 参数、历史）
- `best.pth`: 仅模型权重

训练和推理都必须继续兼容这两种格式。

## 训练损失约定（对称圆柱体）
- 仅监督两项：
  - 平移原点（`pred_t` vs `gt_t`）
  - 旋转矩阵 Y 轴方向（`R[:, :, 1]` 的方向一致性）
- 不再使用完整旋转矩阵逐元素 L1 损失

## 依赖
核心依赖见 `requirements.txt`：
- `torch`, `timm`, `opencv-python`, `numpy`, `pyyaml`, `matplotlib`, `tqdm`, `trimesh`

## 修改规则
1. 优先修改 `efficientpose/` 与 `scripts/`，根目录脚本仅保留兼容包装。
2. 不要在代码中新增硬编码绝对路径，优先使用 CLI 参数。
3. 保持平移单位为米，避免破坏历史模型的可比性。
4. 修改训练流程时，保持 `last.pth` 与 `best.pth` 的保存语义不变。
5. 除非用户明确要求，不要改动 `runs/` 中历史产物。

## 最小验证
每次改动后至少执行：
1. `python3 -m py_compile efficientpose/*.py scripts/*.py tools/*.py train.py infer.py testdataset.py calculate_offset.py 'inference!!.py'`
2. 若有数据集：`python3 scripts/test_dataset.py --data-root <path>`
3. 需要时：`python3 scripts/infer.py --weights <pth> --image <img>`
