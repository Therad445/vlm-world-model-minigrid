# World Model Planning with VLM Scoring in MiniGrid

This repository contains a small MiniGrid experiment. I train a compact RSSM on RGB rollouts and use it for random-shooting MPC. One planner variant additionally scores decoded rollout frames with CLIP.

The CLIP score is computed on future frames decoded from the learned model rollout.

## Methods

The evaluation compares three policies:

| Method | Description |
|---|---|
| `random` | Uniformly random actions. |
| `wm` | Random-shooting MPC in the learned RSSM world model, using predicted reward only. |
| `wm_vlm` | The same planner, with an additional CLIP score on decoded imagined future frames. |

## Measured run

The committed `outputs/`, `logs/` and `report/report.pdf` come from a measured
Google Colab run on a Tesla T4 GPU.

Command:

```bash
python run_pipeline.py --methods random wm wm_vlm
```

Protocol:

- environment: `MiniGrid-Empty-6x6-v0`;
- collected dataset: 240 episodes, including 153 successful episodes;
- world model: compact RSSM trained for 25 epochs;
- planner: random shooting, 32 candidates, horizon 12;
- VLM: `openai/clip-vit-base-patch32`;
- evaluation: 25 episodes per method, seeds `[0, 1, 2, 3, 4]`.

Results:

| Method | Episodes | Success rate | Mean return | Mean episode length |
|---|---:|---:|---:|---:|
| Random | 25 | 0.52 | 0.209 | 82.56 |
| WM planning, no VLM | 25 | 0.88 | 0.520 | 52.00 |
| WM planning + VLM | 25 | 0.40 | 0.166 | 86.04 |

The VLM variant did not improve performance in this run. The report treats this
as an observed limitation of applying off-the-shelf CLIP to symbolic MiniGrid
reconstructions: the green goal remains visually clear, while the decoded agent
can become weak or blurry.

## Repository contents

```text
configs/minigrid_empty.yaml        experiment configuration
src/                               implementation
run_pipeline.py                    one-command pipeline
outputs/metrics.csv                aggregate results
outputs/episodes.csv               per-episode results
outputs/training_loss.csv          RSSM training curve values
outputs/plots/                     result plots
outputs/screenshots/               reconstruction and imagined-rollout checks
outputs/gifs/wm_vlm_episode.gif    rollout visualization
logs/full_run.log                  full Colab run log
report/report.pdf                  final PDF report in Russian
report/report_en.pdf               English PDF report
```

Large regenerable files such as `outputs/data/transitions.npz` and
`outputs/checkpoints/rssm.pt` are intentionally not committed. They are produced
again by `python run_pipeline.py`.

## Quick start

Python 3.10+ is recommended. A CUDA GPU is useful for the VLM path, but the
smoke tests can run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python run_pipeline.py
```

Fast check without loading CLIP:

```bash
python run_pipeline.py --fast --methods random wm
```

Fast end-to-end check with the VLM scorer:

```bash
python run_pipeline.py --fast --methods random wm wm_vlm
```

## Implementation notes

- The RSSM uses a CNN encoder, a GRU deterministic state, a diagonal-Gaussian
  stochastic latent state, an RGB decoder, a reward head and a continuation head.
- MPC samples candidate action sequences, rolls them forward in the learned
  world model, scores the rollouts, executes the first action, and replans.
- The VLM scorer uses one positive text prompt and three negative prompts. It
  scores decoded future frames at imagined steps 3, 6, 9 and 12.
- CLIP features are normalized before the softmax objective is computed.

## Main limitations

- MiniGrid frames are symbolic and low-resolution, while CLIP is trained mostly
  on natural images.
- Decoded rollouts can blur the red agent, which makes the relation
  "agent on goal" hard to score visually.
- Random shooting is transparent but brittle when the objective is noisy.
- A stronger next step would be CEM or a small contrastive scorer trained on
  MiniGrid relation labels.
