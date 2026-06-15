# Project 4 展示提纲

建议总时长：15 分钟展示 + 3 分钟 Q&A。

## Slide 1: 标题与任务

- 题目：基于 Grounding DINO 的开放词汇目标检测复现与评估
- 任务：使用文本 prompt 检测图像目标，并在 COCO 子集上进行量化评估
- 说明本项目属于 Topic 4，可申请 bonus

预计 1 分钟。

## Slide 2: 问题背景

- 传统检测模型依赖固定类别
- 开放词汇检测可以使用任意自然语言类别或短语
- 主要难点：视觉-语言对齐、prompt 敏感性、标签映射、小目标和密集目标

预计 1.5 分钟。

## Slide 3: Related Work

- CLIP：图文对比学习
- OWL-ViT：文本查询式检测
- GLIP：将检测统一成 grounding
- Grounding DINO：结合 DINO detector 和 grounded pre-training

预计 1.5 分钟。

## Slide 4: System Pipeline

- 输入：图片 + prompt
- Grounding DINO 推理
- 输出：bbox、label、score、可视化图
- 转换为 COCO detection JSON
- 使用 COCOeval 计算 AP/AP50/AP75/AR

预计 2 分钟。

## Slide 5: Dataset

- COCO val2017 子集
- 20 类：person、car、dog、cat、chair、laptop 等
- 100 张图，660 个标注框
- 保留 COCO annotation 格式，便于标准评估

预计 1 分钟。

## Slide 6: Baseline

- multi-class prompt：一次输入 20 个类别
- 阈值：box=0.25, text=0.25
- 100 张图结果：AP 0.1970, AP50 0.2681, AR100 0.3142
- 问题：有 118 个 unmatched predictions，说明标签混合影响评估

预计 2 分钟。

## Slide 7: Improvement

- per-class prompt：每次只检测一个类别
- per-label NMS：IoU threshold = 0.5
- 改进结果：AP 0.2355, AP50 0.2832, AR100 0.5229
- unmatched predictions 从 118 降到 0
- 代价：推理更慢，预测框更多，false positive 增加

预计 2 分钟。

## Slide 8: Threshold and Per-Class Analysis

- 展示 `report/figures/threshold_comparison_100.png`
- 展示 `report/figures/per_class_ap_100_perclass_nms.png`
- 说明 dog、bus、cat、couch 等类别提升明显
- 说明 laptop、car、book、chair 等类别仍不稳定

预计 1.5 分钟。

## Slide 9: Failure Cases

- 展示 `report/figures/baseline_false_negative_case.jpg`
- 展示 `report/figures/perclass_false_positive_case.jpg`
- 主要失败类型：
  - 小目标漏检
  - 密集场景重复框
  - 相似类别混淆
  - recall 提升后 precision 下降

预计 1.5 分钟。

## Slide 10: Conclusion

- 完成了模型复现、COCO 子集构建、量化评估、prompt 改进和失败案例分析
- 最好方法将 AP 从 0.1970 提升到 0.2355，AR100 从 0.3142 提升到 0.5229
- 未来方向：Grounding DINO base、大规模图片、类别自适应阈值、跨类别 NMS、自采图片 qualitative demo

预计 1 分钟。

## Q&A Preparation

**Q: 为什么 AP 看起来不高？**  
A: 这是 zero-shot 开放词汇检测，没有对 COCO 子集重新训练；同时 COCOeval 对类别和定位都很严格，IoU=0.50:0.95 的 AP 比 AP50 更难。项目重点是复现、评估和分析，而不是重新训练模型。

**Q: 为什么 per-class prompt 能提升？**  
A: multi-class prompt 容易产生混合文本标签，导致类别映射失败。per-class prompt 每次只查询一个类别，减少文本标签混乱，提高召回。

**Q: per-class prompt 的缺点是什么？**  
A: 推理速度约随类别数线性增加；候选框更多，false positive 增加，需要更好的阈值或 NMS。

**Q: 为什么不用 Grounding DINO 原仓库？**  
A: 本机 Python 3.13 下原仓库自定义算子编译风险较高。HuggingFace Transformers 接口更稳定，也满足复现和评估目标。

**Q: 如果继续做，最优先改进什么？**  
A: 首先跑 Grounding DINO base 的 100 张实验；其次做类别自适应阈值和跨类别 NMS；最后补充自采图片 qualitative demo。
