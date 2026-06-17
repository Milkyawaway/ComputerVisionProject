# Papers

This directory stores project reference papers that are directly used by the implementation and report.

## Grounding DINO

- File: `grounding_dino_arxiv_2303.05499.pdf`
- Paper: **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection**
- arXiv: https://arxiv.org/abs/2303.05499
- PDF source: https://arxiv.org/pdf/2303.05499

The project uses this paper as the main method reference. The local implementation does not retrain or reimplement the full Grounding DINO architecture from scratch; it uses the HuggingFace pretrained model interface and focuses on pipeline reproduction, COCO-style evaluation, prompt strategy experiments, and failure analysis.
