# Grading Policy Checklist

本文件对照 `cv_final_project_2026` 中的 grading policy，说明当前仓库已经覆盖的内容和仍需人工补充的信息。

## Report (60)

| Rubric item | Points | Current coverage |
|---|---:|---|
| Introduction | 10 | `report/report_draft_zh.pdf` 第 1 节说明开放词汇目标检测任务、项目目标和挑战。 |
| Related work | 5 | 第 2 节覆盖 CLIP、OWL-ViT、GLIP、Grounding DINO 和 COCO。 |
| Approach | 10 | 第 3 节说明 Grounding DINO 推理、prompt 规范化、per-class prompt 和 NMS。 |
| Experimental results | 20 | 第 4 节包含 COCO 20 类子集、AP/AP50/AP75/AR 指标、threshold 实验、per-class prompt 改进、每类 AP 和失败案例。 |
| Conclusion | 5 | 第 5 节总结主要发现、限制和未来改进方向。 |
| References | 5 | Reference 部分列出相关论文、COCO 和 HuggingFace 文档。 |
| Overall clarity | 5 | README、报告、结果摘要和复现命令已整理为自包含结构。 |
| Member contribution | add/deduct | 报告末尾已改成带百分比的贡献格式；真实姓名、学号和百分比需要组内最终确认后替换占位。 |

## Presentation (40)

已新增 `presentation/presentation_outline_zh.md`，包含 15 分钟展示结构、每页重点和 Q&A 准备。还需要根据该 outline 制作最终 PPT 或直接作为展示讲稿基础。

## Topic Bonus (10)

项目选择 Topic 4: Open-Vocabulary Object Detection and Visual Grounding，属于 bonus topic。当前仓库完成了模型复现、公开数据集子集评估和改进分析，满足申请 bonus 的基本前提。

## Remaining Manual Items

1. 将报告中的 `待补充姓名`、`待补充学号`、`Name1/Name2/Name3` 替换为真实组员信息。
2. 如果课程要求严格文件名，将 `report/report_draft_zh.pdf` 复制或重命名为 `groupid_final.pdf`。
3. 根据 `presentation/presentation_outline_zh.md` 制作最终展示 slides。
4. 如有自采图片，可放入 `data/custom_demo/images/` 并补充 qualitative demo。
