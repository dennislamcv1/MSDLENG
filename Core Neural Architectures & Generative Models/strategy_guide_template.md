# Transfer Learning Strategy Selection Guide

**Word count target: 250–350 words total.**

Use the four sections below as your structure. Replace each bracketed
prompt with your own analysis — do not leave any section blank, and do
not paste code. This should read as a written recommendation to your
team lead, not as a notebook walkthrough.

---

## Dataset Characteristics

[Briefly describe the dataset's constraints — size (2,400 images),
domain (manufacturing surface defects), and class count (4: scratch,
dent, stain, crack). Explain why these constraints favor one transfer
learning strategy over another.]

## Performance Summary

[Summarize the key metrics — validation accuracy, GPU memory, and
training time — from your comparison table, for the top two performing
conditions. Cite specific numbers from your completed table.]

## Recommended Strategy

[State your recommended architecture and training strategy — one of
the four conditions: ResNet-50 Frozen, ResNet-50 Fine-tuned, ConvNeXt-
Tiny Frozen, or ConvNeXt-Tiny Fine-tuned. Justify the choice with at
least two pieces of performance evidence, and explain why it best fits
the dataset size and compute budget described in the scenario.]

## Augmentation Justification

[Explain why the Albumentations pipeline was configured as it was for
this dataset. Specifically address:
- Why A.CoarseDropout is appropriate for defect classification
- Why A.ColorJitter addresses realistic variation in this dataset
- Why ImageNet normalization (A.Normalize with the standard mean/std
  values) is mandatory when using pretrained TorchVision weights —
  not just that it's required, but what happens mechanically if it's
  omitted.]

---

*Delete this instructions block and the bracketed prompts above before
submitting. Your final guide should be plain prose organized under the
four section headings, totaling 250–350 words.*
