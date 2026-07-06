from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .build_report_en import build as build_en
from .build_report_ru import build_pdf as build_ru
from .utils import load_config


def build(config: dict, lang: str = "ru") -> Path:
    """Build report PDFs.

    lang="ru" builds Russian main report/report.pdf and report/report_ru.pdf.
    lang="en" builds English main report/report.pdf and report/report_en.pdf.
    lang="both" builds both language copies and keeps Russian as report/report.pdf.
    """
    report_dir = Path("report")
    report_dir.mkdir(parents=True, exist_ok=True)

    if lang == "en":
        en_path = build_en(config)
        main = report_dir / "report.pdf"
        shutil.copyfile(en_path, main)
        return main

    if lang == "ru":
        ru_path = build_ru()
        main = report_dir / "report.pdf"
        shutil.copyfile(ru_path, main)
        return main

    if lang == "both":
        build_en(config)
        ru_path = build_ru()
        main = report_dir / "report.pdf"
        shutil.copyfile(ru_path, main)
        return main

    raise ValueError(f"unknown report language: {lang}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minigrid_empty.yaml")
    parser.add_argument("--lang", choices=["ru", "en", "both"], default="both")
    args = parser.parse_args()
    build(load_config(args.config), lang=args.lang)


if __name__ == "__main__":
    main()
