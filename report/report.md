# World-model MPC with a VLM scorer in MiniGrid

Measured report for `MiniGrid-Empty-6x6-v0`.

## Run

- Hardware: Google Colab Tesla T4 GPU
- Command: `python run_pipeline.py --methods random wm wm_vlm`
- Dataset: 240 collected episodes, 153 successful
- Training: compact RSSM, 25 epochs
- Evaluation: 25 episodes per method, seeds `[0, 1, 2, 3, 4]`, five episodes per seed
- Planner: random-shooting MPC, 32 candidates, horizon 12
- VLM: `openai/clip-vit-base-patch32`, applied to decoded imagined future frames at steps 3, 6, 9, 12

## Results

| Method | Episodes | Success rate | Mean return | Mean episode length |
|---|---:|---:|---:|---:|
| Random | 25 | 0.52 | 0.209 | 82.56 |
| WM planning, no VLM | 25 | 0.88 | 0.520 | 52.00 |
| WM planning + VLM | 25 | 0.40 | 0.166 | 86.04 |

The VLM path is implemented and evaluated on future imagined frames, but in this run it did not improve performance over reward-only world-model planning. This is a useful negative result: CLIP is a natural-image VLM and its scores on symbolic MiniGrid reconstructions are noisy.

## Observed failure modes

- CLIP often appears to reward the visible green goal square rather than the relation "agent on the goal".
- The RSSM reconstructs the grid and goal reasonably, but the red agent becomes weak/blurry in reconstructions and imagined frames.
- Since the VLM is applied to decoded model predictions, any decoder error is amplified by the scorer.
- Random shooting with only 32 candidates is unstable when the objective is noisy.

## Future work

- Replace random shooting with CEM.
- Fine-tune a small contrastive MiniGrid scorer on relation labels.
- Add uncertainty penalties to imagined rollouts.
- Compare the VLM scorer against a symbolic oracle scorer for diagnosis only.
- Move from MPC-only planning to a Dreamer actor-critic.
