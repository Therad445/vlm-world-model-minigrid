# Experiment log

## Full run

- Date: 2026-07-05
- Runtime: Google Colab GPU
- GPU: Tesla T4, 15360 MiB VRAM
- Command: `python run_pipeline.py --methods random wm wm_vlm`
- Wall-clock time: about 22 minutes
- Collected data: 240 episodes, 153 successful
- RSSM training: 25 epochs
- Evaluation seeds: `[0, 1, 2, 3, 4]`
- Episodes per method: 25 total, five per seed
- VLM checkpoint: `openai/clip-vit-base-patch32`

## Results

| method | episodes | success_rate | mean_return | mean_episode_length |
|---|---:|---:|---:|---:|
| random | 25 | 0.52 | 0.20896 | 82.56 |
| wm | 25 | 0.88 | 0.52000 | 52.00 |
| wm_vlm | 25 | 0.40 | 0.16564 | 86.04 |

## Observations

The reward-only world-model planner was the strongest policy in this run.
The VLM path was implemented and evaluated, but it degraded performance.

The reconstruction and imagined-rollout screenshots give a plausible reason:
the grid and the green goal remain visible, while the red agent is much less
stable in decoded frames. CLIP can therefore respond to the presence of the
visible green square rather than to the exact spatial relation between the agent
and the goal.

I do not claim that the VLM scorer improves the planner in this run. The experiment shows that the mechanism works technically, but CLIP is not well calibrated for decoded MiniGrid frames.

## Future work

- Replace random shooting with CEM.
- Fine-tune a small contrastive scorer on MiniGrid relation labels.
- Add uncertainty penalties to imagined states.
- Compare with a symbolic oracle scorer for debugging only.
- Try an object-centric world model so that the agent position is preserved more
  reliably in decoded frames.
