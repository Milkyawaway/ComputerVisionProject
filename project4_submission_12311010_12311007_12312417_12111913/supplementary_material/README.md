# Supplementary Material

该目录包含 Project 4: Open-Vocabulary Object Detection and Visual Grounding 的补充材料。补充材料只保留报告和展示中需要引用的轻量结果，不包含完整 COCO 图片、完整推理输出或模型缓存。

## Contents

```text
supplementary_material/
  README.md
  summary.csv
  summary.json
  report_figures/
    threshold_comparison_1000_base.png
    prompt_comparison_1000_base.png
    per_class_ap_1000_base_multiclass_t025.png
    base1000_multiclass_false_negative_case.jpg
    base1000_perclass_false_positive_case.jpg
```

## Files

| File | Description |
|---|---|
| `summary.csv` | 1000 张 COCO 子集上 Grounding DINO base 的主要实验结果表格 |
| `summary.json` | 与 `summary.csv` 对应的结构化结果，方便复查实验配置和指标 |
| `report_figures/threshold_comparison_1000_base.png` | 不同 threshold 下 AP、AP50、AP75、AR100 对比图 |
| `report_figures/prompt_comparison_1000_base.png` | multi-class prompt、per-class prompt、score cutoff 的结果对比图 |
| `report_figures/per_class_ap_1000_base_multiclass_t025.png` | 最佳 AP 配置下 20 个 COCO 类别的 per-class AP |
| `report_figures/base1000_multiclass_false_negative_case.jpg` | multi-class prompt 的 false negative 失败案例 |
| `report_figures/base1000_perclass_false_positive_case.jpg` | per-class prompt + NMS 的 false positive 失败案例 |

## Main Experiment

主要实验设置：

- 模型：`IDEA-Research/grounding-dino-base`
- 数据集：COCO val2017 20 类子集
- 图片数量：1000
- 标注框数量：5892
- 主评估工具：`pycocotools.COCOeval`
- 主配置：multi-class prompt, `box_threshold=0.25`, `text_threshold=0.25`

主要结果：

| Method | AP | AP50 | AP75 | AR100 | Valid predictions | Unmatched predictions |
|---|---:|---:|---:|---:|---:|---:|
| multi-class prompt, 0.40 / 0.30 | 0.0732 | 0.0988 | 0.0820 | 0.0883 | 1263 | 10 |
| multi-class prompt, 0.25 / 0.25 | 0.2073 | 0.2975 | 0.2289 | 0.3073 | 5924 | 731 |
| multi-class prompt, 0.20 / 0.20 | 0.1804 | 0.2780 | 0.1924 | 0.3023 | 11272 | 2188 |
| per-class prompt + NMS | 0.1284 | 0.1592 | 0.1427 | 0.4478 | 29706 | 0 |
| per-class + NMS + score cutoff 0.30 | 0.1225 | 0.1500 | 0.1364 | 0.3997 | 20956 | 0 |
| per-class + NMS + score cutoff 0.35 | 0.1174 | 0.1420 | 0.1308 | 0.3578 | 14872 | 0 |
| per-class + NMS + score cutoff 0.40 | 0.1122 | 0.1348 | 0.1252 | 0.3239 | 10321 | 0 |

## How to Interpret

- `AP` 是 COCO 主指标，表示 IoU 0.50 到 0.95 的平均精度。
- `AP50` 和 `AP75` 分别表示 IoU 阈值为 0.50 和 0.75 时的平均精度。
- `AR100` 表示每张图最多保留 100 个检测框时的平均召回率。
- `Valid predictions` 是成功映射到 COCO 20 类并进入 COCOeval 的预测数。
- `Unmatched predictions` 是模型输出了文本标签，但无法精确映射到 COCO 类别的预测数。

## Summary

本实验中，`multi-class prompt, 0.25 / 0.25` 取得最高 AP。降低 threshold 可以提高召回，但过低会产生更多误检和 unmatched labels。`per-class prompt + NMS` 可以消除 unmatched labels 并提高 AR100，但预测框数量显著增加，false positive 增多，因此 AP 下降。
