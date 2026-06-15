# Custom Demo Images

Put self-collected classroom, campus, or lab images in `data/custom_demo/images/`.

Recommended command:

```bash
python scripts/infer.py \
  --image-dir data/custom_demo/images \
  --prompt-file configs/custom_demo_prompt.txt \
  --model-id IDEA-Research/grounding-dino-base \
  --box-threshold 0.25 \
  --text-threshold 0.25 \
  --per-class-prompts \
  --nms-iou-threshold 0.5 \
  --output-dir outputs/custom_demo
```
