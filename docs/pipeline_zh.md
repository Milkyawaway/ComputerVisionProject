# 当前 Pipeline 说明

本项目复现的是 **open-vocabulary object detection pipeline**，即输入图像和自然语言文本 prompt，输出目标框、文本标签、置信度和可视化结果，并在 COCO 子集上进行标准检测指标评估。

## 1. Reproduction Scope

本项目不是从零训练 Grounding DINO，也不是手写完整模型结构。我们采用 Grounding DINO 的公开预训练模型，重点复现和扩展完整实验流程。

| 部分 | 当前实现方式 |
|---|---|
| Grounding DINO 模型结构 | 直接使用 HuggingFace `AutoModelForZeroShotObjectDetection.from_pretrained` |
| 图像/文本预处理 | 直接使用 HuggingFace `AutoProcessor.from_pretrained` |
| 模型基础后处理 | 调用 `processor.post_process_grounded_object_detection` |
| 单图/URL/批量推理 CLI | 本项目实现，见 `scripts/infer.py` |
| prompt 规范化 | 本项目实现，见 `src/ovd/grounding_dino.py` |
| per-class prompt 策略 | 本项目实现，见 `scripts/infer.py` |
| per-label NMS 和 score cutoff | 本项目实现，见 `scripts/infer.py` |
| COCO 子集构建 | 本项目实现，见 `scripts/prepare_coco_subset.py` |
| COCO detection 格式转换 | 本项目实现，见 `scripts/eval_coco.py` |
| AP/AP50/AP75/AR 和 per-class AP | 本项目基于 `pycocotools` 实现评估封装 |
| threshold 对比图和失败案例可视化 | 本项目实现 |

因此，当前工作的定位是：**基于开源预训练模型复现开放词汇检测流程，并完成数据准备、推理封装、量化评估、prompt 改进和错误分析**。

## 2. Overall Data Flow

整体数据流如下：

```text
COCO val2017 / custom images
        |
        v
prepare_coco_subset.py
        |
        v
images + COCO annotations + prompt file
        |
        v
infer.py
        |
        v
GroundingDinoDetector
        |
        v
AutoProcessor + AutoModelForZeroShotObjectDetection
        |
        v
boxes_xyxy + scores + text_labels
        |
        v
NMS / score cutoff / visualization / JSON output
        |
        v
eval_coco.py
        |
        v
COCOeval metrics + per-class AP + unmatched labels
        |
        v
export_failure_cases.py
        |
        v
false positive / false negative analysis
```

## 3. Data Preparation

脚本：`scripts/prepare_coco_subset.py`

该脚本负责从 COCO val2017 中构建一个适合课程项目规模的开放词汇检测子集。默认类别为 20 类：

```text
person, bicycle, car, motorcycle, bus, truck, traffic light,
dog, cat, horse, chair, couch, dining table, bottle, cup,
laptop, book, backpack, umbrella, cell phone
```

输出内容包括：

- `annotations/instances_val2017_subset.json`：保留 COCO annotation 格式；
- `images/`：所选图片；
- `metadata/subset_manifest.csv`：子集图片清单；
- `prompts/coco_20_classes.txt`：Grounding DINO 使用的 prompt 类别文件。

保留 COCO 格式的原因是后续可以直接使用 `pycocotools.COCOeval` 计算标准目标检测指标。

## 4. Prompt Processing

代码位置：`src/ovd/grounding_dino.py`

Grounding DINO 对英文、低写、句号分隔的 prompt 更稳定，因此项目中会把输入 prompt 规范化。例如：

```text
person, car, traffic light
```

会被转换为：

```text
a person. a car. a traffic light.
```

如果 prompt 文件中每行是一个类别，`infer.py` 会读取所有类别并拼接为 Grounding DINO 推荐的句号分隔格式。

## 5. Model Inference

脚本：`scripts/infer.py`

模型封装：`src/ovd/grounding_dino.py`

推理入口支持三种输入：

- `--image`：单张本地图片；
- `--image-url`：单张网络图片；
- `--image-dir`：目录批量推理。

核心模型调用为：

```python
AutoProcessor.from_pretrained(model_id)
AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
processor.post_process_grounded_object_detection(...)
```

模型输出会统一整理为：

- `boxes_xyxy`：检测框，格式为 `[x0, y0, x1, y1]`；
- `scores`：置信度；
- `text_labels`：模型输出或当前 prompt 对应的文本标签；
- `prompt`、`model_id`、`threshold`、`device` 等元数据。

每张图片会生成两个文件：

- `<image_stem>.json`：结构化预测结果；
- `<image_stem>_ovd.jpg`：带 bbox、label、score 的可视化图。

## 6. Multi-Class Prompt vs Per-Class Prompt

项目实现了两种推理策略。

### Multi-Class Prompt

一次性输入 20 个类别：

```text
a person. a bicycle. a car. ... a cell phone.
```

优点是速度快，每张图片只需一次模型前向传播。缺点是模型可能输出混合标签，例如 `traffic light an umbrella`，导致后续 COCO 类别映射失败。

### Per-Class Prompt

每次只输入一个类别：

```text
a person.
a car.
a traffic light.
```

然后将所有类别结果合并。优点是标签更干净，unmatched prediction 明显减少，recall 更高。缺点是速度约随类别数线性增加，并且会产生更多候选框和 false positive。

当前 1000 张 `grounding-dino-base` 实验中，multi-class prompt, 0.25 / 0.25 是 AP 最优方案；per-class prompt + NMS 的 AR100 更高、未匹配标签为 0，但 AP 更低：

| Method | AP | AP50 | AP75 | AR100 | Unmatched predictions |
|---|---:|---:|---:|---:|---:|
| multi-class prompt, 0.25 / 0.25 | 0.2073 | 0.2975 | 0.2289 | 0.3073 | 731 |
| per-class prompt + NMS | 0.1284 | 0.1592 | 0.1427 | 0.4478 | 0 |

## 7. Post-Processing

脚本：`scripts/infer.py`

项目在 HuggingFace 的基础后处理之后额外实现了两步：

1. **per-label NMS**：同一类别内，如果两个框 IoU 大于阈值，例如 0.5，则保留置信度更高的框。
2. **post-score cutoff**：在 NMS 后再按 score 做筛选，用于 threshold sweep。

典型命令：

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

## 8. COCO-Style Evaluation

脚本：`scripts/eval_coco.py`

评估脚本做了以下转换：

1. 读取 `infer.py` 生成的每张图 JSON；
2. 根据图片文件名 stem 映射到 COCO `image_id`；
3. 将预测框从 `xyxy` 转为 COCO detection 所需的 `xywh`；
4. 将 `text_labels` 归一化，例如去掉 `a`、`an`、`the`；
5. 与 COCO 20 类类别名精确匹配；
6. 无法匹配的预测写入 `unmatched_predictions.csv`；
7. 使用 `pycocotools.COCOeval` 计算 AP、AP50、AP75、AR 等指标；
8. 输出 summary CSV/JSON、per-class AP CSV 和图表。

主要输出包括：

- `predictions_coco.json`
- `metrics_summary.json`
- `metrics_summary.csv`
- `per_class_ap.csv`
- `per_class_ap.png`
- `unmatched_predictions.csv`

## 9. Failure Case Analysis

脚本：`scripts/export_failure_cases.py`

该脚本把预测框与 ground truth 做 IoU 匹配，并导出 false positive 和 false negative 最多的样例图。可视化颜色约定为：

- 绿色：ground truth；
- 蓝色：true positive；
- 红色：false positive；
- 橙色：false negative。

该部分主要用于分析模型在哪些场景失败，例如小目标漏检、密集场景重复框、相似类别混淆，以及 per-class prompt 提高 recall 后引入更多 false positive。

## 10. Pipeline Summary

当前 pipeline 的核心贡献不是重新训练一个新模型，而是把 Grounding DINO 预训练模型接入一个完整、可复现、可评估的开放词汇检测实验系统。它覆盖：

- 模型复现和推理封装；
- COCO 子集构建；
- prompt 策略实验；
- NMS 和 score cutoff 后处理；
- COCO 标准指标评估；
- per-class AP 和 threshold comparison；
- false positive / false negative 可视化分析；
- 报告和 presentation 所需的结果摘要。

这个范围与课程要求中的 “start from existing open-source models and focus on reproduction, evaluation, and analysis” 是一致的。
