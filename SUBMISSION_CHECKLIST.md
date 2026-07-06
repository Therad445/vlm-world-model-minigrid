# Submission checklist

Before submitting the repository link, check that these files are present:

- [ ] `README.md`
- [ ] `configs/minigrid_empty.yaml`
- [ ] `src/`
- [ ] `run_pipeline.py`
- [ ] `report/report.pdf`
- [ ] `outputs/metrics.csv`
- [ ] `outputs/episodes.csv`
- [ ] `outputs/training_loss.csv`
- [ ] `outputs/plots/success_rate.png`
- [ ] `outputs/plots/training_loss.png`
- [ ] `outputs/screenshots/reconstruction.png`
- [ ] `outputs/screenshots/imagined_rollout.png`
- [ ] `outputs/gifs/wm_vlm_episode.gif`
- [ ] `logs/full_run.log`

Recommended final check:

```bash
python -m compileall src run_pipeline.py
cat outputs/metrics.csv
```

Do not commit `.venv/`, `__pycache__/`, `outputs/data/` or
`outputs/checkpoints/`.
