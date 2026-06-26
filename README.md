# AT-STGCN Skeleton Sign Language Recognition

本仓库对应论文 `AT-STGCN Skeleton Sign Language Recognition`，实现面向孤立词手语识别的
AT-STGCN（Adaptive Topology Spatial-Temporal Graph Convolutional Network）。
当前公开复现实验以 68 点紧凑骨架序列为输入，不再以旧图像化骨架方案作为主路径。

## 当前主线

1. 使用 MediaPipe Holistic 从视频中抽取 68 点手语骨架。
2. 将样本保存为 `(T, 68, 3)` 的 `.npy` 骨架序列。
3. 训练 skeleton-only AT-STGCN：多源骨架特征、固定多跳拓扑、自适应邻接、边重要性、部位池化和 ArcFace 分类头。
4. 使用统一验证/评估脚本导出 Top-1、Top-5、混淆矩阵和分类报告。

论文主配置：

| 数据集 | 配置 |
|---|---|
| AUTSL | `configs/autsl_skeleton.yaml` |
| ASL Citizen | `configs/asl_citizen_skeleton.yaml` |
| SAM-style 27 点扩展 | `configs/sam27_autsl.yaml`, `configs/sam27_asl_citizen.yaml` |

## 安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 数据清单

视频清单：

```csv
video_path,label,split
/path/to/video_0001.mp4,book,train
/path/to/video_0002.mp4,book,val
/path/to/video_0003.mp4,book,test
```

骨架清单：

```csv
keypoints_path,path,label,split,valid_pose_frames
data/autsl_skeleton/train/0/sample.npy,data/autsl_skeleton/train/0/sample.npy,0,train,64
```

其中 `.npy` 数组形状为 `(T, 68, 3)`；训练时会被整理为 PyTorch 的 `(3, T, 68)`。

## 生成清单和骨架

AUTSL：

```bash
python scripts/create_autsl_manifest.py \
  --autsl-root /path/to/AUTSL \
  --output data/autsl_videos_manifest.csv \
  --modality color

python scripts/preprocess_videos.py \
  --input-manifest data/autsl_videos_manifest.csv \
  --output-dir data/autsl_skeleton \
  --output-manifest data/autsl_skeleton_manifest.csv \
  --min-valid-pose-frames 8 \
  --fallback-full-video-on-short
```

ASL Citizen：

```bash
python scripts/create_asl_citizen_manifest.py \
  --asl-root /path/to/ASL_Citizen \
  --splits-dir /path/to/ASL_Citizen/splits \
  --videos-dir /path/to/ASL_Citizen/videos \
  --output data/asl_citizen_videos_manifest.csv

python scripts/preprocess_videos.py \
  --input-manifest data/asl_citizen_videos_manifest.csv \
  --output-dir data/asl_citizen_skeleton \
  --output-manifest data/asl_citizen_skeleton_manifest.csv \
  --min-valid-pose-frames 8 \
  --fallback-full-video-on-short
```

如果已有关键点数组，可用：

```bash
python scripts/preprocess_skeletons.py \
  --input-manifest keypoints_manifest.csv \
  --output-dir data/my_skeleton \
  --output-manifest data/my_skeleton_manifest.csv \
  --source array
```

## 训练

```bash
python scripts/train.py --config configs1/autsl_skeleton.yaml
python scripts/train.py --config configs1/asl_citizen_skeleton.yaml
```

主要实验、对比实验和消融实验可用统一入口检查或运行：

```bash
# 只打印论文已有实验命令：主实验、顺序消融、时序增强、配置敏感性、复杂度边界和 SAM27
python scripts/run_paper_experiments.py --stage train --dry-run

# 只打印 configs1/repeated_seeds 下的随机种子实验命令
python scripts/run_paper_experiments.py --suite repeated_seed --stage train --dry-run

# 实际运行 AUTSL 的主实验和顺序消融
python scripts/run_paper_experiments.py --datasets autsl --suite main,ablation --stage train

# 实际运行论文中的拓展实验：时序增强、统一配置敏感性、特征融合与关系复杂度边界
python scripts/run_paper_experiments.py --suite temporal,sensitivity,complexity --stage train

# 对已有标准 AT-STGCN checkpoint 运行验证集评估
python scripts/run_paper_experiments.py --suite main,ablation,temporal,sensitivity --stage eval --split val
```

训练输出位于配置里的 `output_dir`，通常包含：

```text
best.pt
last.pt
soup.pt
model_soup.json
label_map.json
history.csv
history.json
config_used.yaml
```

## 评估

```bash
python scripts/evaluate.py \
  --model runs/autsl_skeleton_stgcn_v5_arcface/best.pt \
  --manifest data/autsl_skeleton_manifest.csv \
  --label-map runs/autsl_skeleton_stgcn_v5_arcface/label_map.json \
  --split val \
  --image-height 64 \
  --output-dir runs/paper_evaluation/AUTSL/main_val \
  --tta-flip \
  --tta-scales 0.95,1.0,1.05 \
  --require-paper-valid
```

单视频预测：

```bash
python scripts/predict_video.py \
  --video /path/to/sign.mp4 \
  --model runs/autsl_skeleton_stgcn_v5_arcface/best.pt \
  --label-map runs/autsl_skeleton_stgcn_v5_arcface/label_map.json \
  --height 64 \
  --top-k 5
```

## 代码结构

```text
sl_atstgcn/
  at_stgcn.py               # 当前论文公开 API，指向 skeleton-only AT-STGCN
  model.py                  # skeleton-only AT-STGCN 模型实现
  data.py                   # manifest、骨架加载、缺失点修复和数据增强
  graph.py                  # 68 点骨架拓扑、镜像索引和 DFS 元数据
  keypoints.py              # MediaPipe 到 68 点骨架的映射
  extractor.py              # MediaPipe Holistic 抽取器
  evaluation_protocol.py    # 论文评估协议审计

scripts/
  train.py                  # skeleton-only AT-STGCN 训练入口
  evaluate.py               # skeleton-only AT-STGCN 评估入口
  predict_video.py          # 单视频预测
  preprocess_videos.py      # 视频到骨架序列
  preprocess_skeletons.py   # 已有数组到统一骨架序列
```

## 清理状态

当前公开入口已经收敛到 skeleton-only AT-STGCN：训练、评估、视频预处理和实验套件均使用
68 点骨架序列，不再包含旧 CNN 图像分支。`sl_atstgcn.graph` 中保留 `DFS_*` 元数据，
用于解释论文拓扑顺序并保持拓扑测试稳定。
