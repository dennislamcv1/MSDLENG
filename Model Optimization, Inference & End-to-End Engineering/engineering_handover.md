# Engineering Handover: Contoso Fraud Detection Model

**Role:** Lead MLOps Architect · **Date:** 16 October 2026 · **Status:** Green deployment at 10% traffic

## 1. Model Card

| Field | Detail |
|---|---|
| Model | 2B-parameter transformer, binary sequence classifier (fraud / legitimate) |
| Intended use | Real-time scoring of card transactions; high scores route to analyst review, not automatic account action |
| Training | DeepSpeed ZeRO-3, bf16, 2 × Standard_ND96amsr_A100_v4 (16 A100 GPUs, InfiniBand); 12-trial Random sweep over learning rate and weight decay, BanditPolicy (slack 0.1) |
| Artifact | ONNX graph, dynamic INT8 (Optimum AVX2 config, per-channel weights) |
| Serving | `fraud-detection-endpoint`, Microsoft Entra token auth; green = 2 × Standard_NC4as_T4_v3; providers TensorRT → CUDA → CPU |
| Metrics | Baseline 98.5% / 250 ms; INT8 97.8% / 45 ms (precomputed benchmark) |
| Limitations | Accuracy is the only benchmarked metric, so fraud-class recall must be measured before 100% rollout. Dynamic INT8 matrix multiplies run outside TensorRT; static calibration is the follow-up that moves them onto TensorRT INT8. |

## 2. Latency–Cost–Accuracy Trade-off Matrix

| Metric | Baseline PyTorch | ONNX INT8 | Change |
|---|---|---|---|
| Accuracy | 98.5% | 97.8% | −0.7 pp |
| Errors per 100,000 transactions | 1,500 | 2,200 | +700 |
| Mean latency | 250 ms | 45 ms | −82% (5.6× faster) |
| 100 ms SLA | Fails by 150 ms | Passes, 55 ms headroom | — |
| Weight memory | 8 GB (FP32) | 2 GB (INT8) | −75% |
| Compute time per 1,000 serial requests | 250 s | 45 s | −82% |

Cost rows assume both models run on the same hardware and that the baseline stores FP32 weights.

**Justification.** The baseline cannot be deployed at any accuracy, because it misses the non-negotiable SLA by 150%. INT8 meets it with 55 ms to spare and cuts compute time per request by 82%, so the same traffic needs roughly one-fifth of the serving capacity. The price is 700 extra misclassifications per 100,000 transactions. We accept that for this rollout, limit exposure through the 10% canary, and plan to recover accuracy with static calibration, which keeps the latency gain.

## 3. Rollout and Rollback Playbook

**Stages:** 90/10 for at least 24 hours, then 50/50 for 24 hours, then 0/100. Blue keeps its full instance count until green has served 100% for seven days.

**Promotion gate:** green's startup log must show TensorRT as the active execution provider.

**Rollback triggers** (any one, measured on green's own metrics):

1. p95 latency above 100 ms for two consecutive 5-minute windows.
2. HTTP 5xx errors plus 408 timeouts above 1% of requests over 5 minutes.
3. Accuracy on the daily labelled replay set below 97.3% (0.5 pp under benchmark).
4. Fraud-class recall more than 1 pp below blue on the same replay set.

**Rollback steps:**

1. Run `python contoso_fraud_pipeline.py --rollback`. One update sets traffic to blue 100 / green 0; blue is already at full capacity, so nothing cold-starts.
2. Confirm blue's p95 latency and error rate return to baseline within 15 minutes.
3. Keep green at 0% for log and Application Insights diagnosis, then fix and redeploy.
4. Log the incident. A new green re-enters at 10% and must clear all four triggers again.
