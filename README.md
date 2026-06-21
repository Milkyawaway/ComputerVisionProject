# Project 4: Open-Vocabulary Object Detection

本项目实现 Computer Vision 2026 Final Project Topic 4: **Open-Vocabulary Object Detection and Visual Grounding**。

项目基于 HuggingFace `transformers` 中的 Grounding DINO 预训练模型，完成开放词汇目标检测的模型复现、COCO 子集构建、批量推理、COCO-style 评测和失败案例分析。代码重点不是从零训练模型，而是复现并整理一个可运行、可评估、可分析的开放词汇检测 pipeline。

## 1. Project Scope

当前代码覆盖课程要求中的两个核心技术部分：

| 要求 | 当前实现 |
|---|---|
| Method Reproduction | 使用 `IDEA-Research/grounding-dino-base` 复现 Grounding DINO 开放词汇检测流程 |
| Dataset Evaluation | 在 COCO val2017 20 类子集上进行 COCOeval 评测 |
| Analysis Support | 支持 threshold 对比、per-class prompt、NMS、score cutoff、per-class AP、失败案例导出 |

主要实验使用：

- 模型：`IDEA-Research/grounding-dino-base`
- 数据：COCO val2017 1000 张图片、20 个类别
- 主配置：multi-class prompt, `box_threshold=0.25`, `text_threshold=0.25`
- 主指标：AP、AP50、AP75、AR100

## 2. Directory Structure

```text
configs/
  coco_20_classes.txt          # COCO 20 类 prompt
  custom_demo_prompt.txt       # 自定义 demo prompt

data/
  README.md                    # 数据目录说明

docs/
  pipeline_zh.md               # 中文 pipeline 说明

scripts/
  infer.py                     # 单图、URL、批量推理
  prepare_coco_subset.py       # 构建 COCO 子集
  eval_coco.py                 # COCO-style 评测
  export_failure_cases.py      # 导出 FP/FN 失败案例

src/ovd/
  grounding_dino.py            # Grounding DINO 模型封装
  visualize.py                 # 检测框可视化

tests/
  test_eval_and_postprocess.py # 评测和后处理单元测试

requirements.txt               # Python 依赖
```

本地生成的 COCO 图片、完整预测输出和模型运行结果通常体积较大，提交代码包时可以不包含；它们可以通过下面命令重新生成。

## 3. Environment

```bash
cd project4_ovd
python -m pip install -r requirements.txt
```

推荐有 CUDA 的环境运行 Grounding DINO。默认推理设备是 `cuda:0`，也可以通过 `--device cpu` 切换到 CPU。

快速检查：

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection"
python -m unittest discover -s tests
```

## 4. Single Image Demo

```bash
python scripts/infer.py \
  --image-url http://images.cocodataset.org/val2017/000000039769.jpg \
  --prompt "a cat. a remote control." \
  --model-id IDEA-Research/grounding-dino-base \
  --box-threshold 0.25 \
  --text-threshold 0.25 \
  --output-dir outputs/demo
```

每张图片会生成：

- `<image_stem>.json`：结构化预测结果，包含 `boxes_xyxy`、`scores`、`text_labels`
- `<image_stem>_ovd.jpg`：带 bbox、label、score 的可视化结果

## 5. Prepare COCO Subset

构建报告中使用的 1000 张 COCO 子集：

```bash
python scripts/prepare_coco_subset.py \
  --max-images 1000 \
  --download-images \
  --output-dir data/coco_subset_1000
```

输出结构：

```text
data/coco_subset_1000/
  annotations/instances_val2017_subset.json
  images/
  metadata/subset_manifest.csv
  prompts/coco_20_classes.txt
```

该 annotation 文件保留 COCO 格式，后续可直接用于 `pycocotools.COCOeval`。

## 6. Batch Inference

主实验 multi-class prompt：

```bash
python scripts/infer.py \
  --image-dir data/coco_subset_1000/images \
  --prompt-file data/coco_subset_1000/prompts/coco_20_classes.txt \
  --model-id IDEA-Research/grounding-dino-base \
  --box-threshold 0.25 \
  --text-threshold 0.25 \
  --output-dir outputs/coco_subset_1000_base_multiclass_t025
```

per-class prompt + NMS：

```bash
python scripts/infer.py \
  --image-dir data/coco_subset_1000/images \
  --prompt-file data/coco_subset_1000/prompts/coco_20_classes.txt \
  --model-id IDEA-Research/grounding-dino-base \
  --box-threshold 0.25 \
  --text-threshold 0.25 \
  --per-class-prompts \
  --nms-iou-threshold 0.5 \
  --output-dir outputs/coco_subset_1000_base_perclass_nms_t025
```

## 7. COCO Evaluation

```bash
python scripts/eval_coco.py \
  --annotations data/coco_subset_1000/annotations/instances_val2017_subset.json \
  --pred-dir outputs/coco_subset_1000_base_multiclass_t025 \
  --output-dir outputs/eval_1000_base_multiclass_t025 \
  --experiment-name base1000_multiclass_t025
```

评测输出包括：

- `predictions_coco.json`：转换后的 COCO detection result
- `metrics_summary.json` / `metrics_summary.csv`：整体 AP、AP50、AP75、AR
- `per_class_ap.csv` / `per_class_ap.png`：每类 AP
- `unmatched_predictions.csv`：无法映射到 COCO 类别的开放文本标签

## 8. Threshold Comparison

先用不同 threshold 分别运行 `infer.py`，再分别运行 `eval_coco.py`。例如：

```bash
python scripts/infer.py \
  --image-dir data/coco_subset_1000/images \
  --prompt-file data/coco_subset_1000/prompts/coco_20_classes.txt \
  --model-id IDEA-Research/grounding-dino-base \
  --box-threshold 0.20 \
  --text-threshold 0.20 \
  --output-dir outputs/coco_subset_1000_base_multiclass_t020
```

多个实验的 summary 可以用 `eval_coco.py --compare` 生成对比表和图：

```bash
python scripts/eval_coco.py \
  --compare \
  outputs/eval_1000_base_multiclass_t020/metrics_summary.json \
  outputs/eval_1000_base_multiclass_t025/metrics_summary.json \
  outputs/eval_1000_base_multiclass_t040_t030/metrics_summary.json \
  --output-dir outputs/eval_1000_base_threshold_compare
```

## 9. Failure Case Visualization

```bash
python scripts/export_failure_cases.py \
  --annotations data/coco_subset_1000/annotations/instances_val2017_subset.json \
  --predictions outputs/eval_1000_base_multiclass_t025/predictions_coco.json \
  --image-dir data/coco_subset_1000/images \
  --output-dir outputs/failure_cases_1000_base_multiclass_t025 \
  --top-k 8
```

可视化颜色约定：

- 绿色：ground truth
- 蓝色：true positive
- 红色：false positive
- 橙色：false negative

## 10. Main Result Summary

1000 张 COCO 子集、Grounding DINO base 的主要结果：

| Method | AP | AP50 | AP75 | AR100 | Valid predictions | Unmatched predictions |
|---|---:|---:|---:|---:|---:|---:|
| multi-class prompt, 0.40 / 0.30 | 0.0732 | 0.0988 | 0.0820 | 0.0883 | 1263 | 10 |
| multi-class prompt, 0.25 / 0.25 | 0.2073 | 0.2975 | 0.2289 | 0.3073 | 5924 | 731 |
| multi-class prompt, 0.20 / 0.20 | 0.1804 | 0.2780 | 0.1924 | 0.3023 | 11272 | 2188 |
| per-class prompt + NMS | 0.1284 | 0.1592 | 0.1427 | 0.4478 | 29706 | 0 |
| per-class + NMS + score cutoff 0.30 | 0.1225 | 0.1500 | 0.1364 | 0.3997 | 20956 | 0 |
| per-class + NMS + score cutoff 0.35 | 0.1174 | 0.1420 | 0.1308 | 0.3578 | 14872 | 0 |
| per-class + NMS + score cutoff 0.40 | 0.1122 | 0.1348 | 0.1252 | 0.3239 | 10321 | 0 |

结论：multi-class prompt, 0.25 / 0.25 是本实验中的 AP 最优配置；per-class prompt + NMS 能提高召回率并消除 unmatched label，但会产生更多 false positive，因此整体 AP 较低。

## 11. Notes

- Grounding DINO 的 prompt 推荐使用英文小写和句号分隔。
- `box_threshold` 决定 query/box 是否保留，`text_threshold` 决定哪些文本 token 被恢复成 label。
- `eval_coco.py` 采用精确类别匹配，混合标签如 `a bicycle a motorcycle` 会写入 `unmatched_predictions.csv`，不参与 COCOeval。
- 本项目评估的是 COCO-style object detection，不包含 RefCOCO 等 visual grounding 专用指标。
