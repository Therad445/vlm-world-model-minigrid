.PHONY: install data train evaluate report all smoke smoke-vlm clean

install:
	python -m pip install -r requirements.txt

data:
	python -m src.collect_data --config configs/minigrid_empty.yaml

train:
	python -m src.train_world_model --config configs/minigrid_empty.yaml

evaluate:
	python -m src.evaluate --config configs/minigrid_empty.yaml --methods random wm wm_vlm

report:
	python -m src.build_report --config configs/minigrid_empty.yaml

all: data train evaluate report

# Fast check that avoids CLIP download/load.
smoke:
	python run_pipeline.py --fast --methods random wm

# Fast check of the complete VLM path; requires transformers + CLIP weights.
smoke-vlm:
	python run_pipeline.py --fast --methods random wm wm_vlm

clean:
	rm -rf outputs report/report.pdf
