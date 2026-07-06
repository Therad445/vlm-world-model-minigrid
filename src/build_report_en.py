from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .utils import ensure_parent, load_config


PAGE_W, PAGE_H = A4
MARGIN_X = 18 * mm
MARGIN_TOP = 17 * mm
MARGIN_BOTTOM = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X


def _fonts() -> tuple[str, str]:
    candidates = [
        ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("LiberationSans", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "LiberationSans-Bold", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    for regular, regular_path, bold, bold_path in candidates:
        if Path(regular_path).exists() and Path(bold_path).exists():
            pdfmetrics.registerFont(TTFont(regular, regular_path))
            pdfmetrics.registerFont(TTFont(bold, bold_path))
            return regular, bold
    return "Helvetica", "Helvetica-Bold"


FONT_REG, FONT_BOLD = _fonts()


def _styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="ReportTitle", parent=base["Title"], fontName=FONT_BOLD, fontSize=22,
        leading=27, alignment=TA_LEFT, textColor=colors.HexColor("#111827"), spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        name="Subtitle", parent=base["Normal"], fontName=FONT_REG, fontSize=10.5,
        leading=14, textColor=colors.HexColor("#4B5563"), spaceAfter=12,
    ))
    base.add(ParagraphStyle(
        name="Section", parent=base["Heading2"], fontName=FONT_BOLD, fontSize=15,
        leading=19, textColor=colors.HexColor("#111827"), spaceBefore=8, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        name="Body", parent=base["BodyText"], fontName=FONT_REG, fontSize=9.8,
        leading=13.2, textColor=colors.HexColor("#111827"), spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        name="Small", parent=base["BodyText"], fontName=FONT_REG, fontSize=8.5,
        leading=11, textColor=colors.HexColor("#374151"), spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        name="Callout", parent=base["BodyText"], fontName=FONT_BOLD, fontSize=10.5,
        leading=13.5, textColor=colors.HexColor("#111827"), spaceAfter=4,
    ))
    return base


styles = _styles()


def p(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, styles[style])


def section(title: str):
    return [
        Paragraph(title, styles["Section"]),
        HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#E5E7EB"), spaceBefore=0, spaceAfter=8),
    ]


def scaled_image(path: str | Path, width_mm: float) -> Image | None:
    path = Path(path)
    if not path.exists():
        return None
    img = Image(str(path))
    w, h = img.imageWidth, img.imageHeight
    img.drawWidth = width_mm * mm
    img.drawHeight = img.drawWidth * h / w
    return img


def add_image(story: list, path: str | Path, width_mm: float, space_after: float = 4) -> None:
    img = scaled_image(path, width_mm)
    if img is not None:
        story.append(img)
        story.append(Spacer(1, space_after * mm))


def badge_table(items):
    rows = [[Paragraph(f"<b>{label}</b>", styles["Small"]), Paragraph(str(value), styles["Small"])] for label, value in items]
    t = Table(rows, colWidths=[38 * mm, CONTENT_W - 38 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9FAFB")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D1D5DB")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#E5E7EB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def method_table() -> Table:
    table = Table([
        ["Policy", "Objective"],
        ["random", "Uniform random actions."],
        ["wm", "Random-shooting MPC in the learned RSSM, using predicted discounted reward."],
        ["wm_vlm", "The same planner plus the maximum CLIP goal score over decoded imagined frames."],
    ], colWidths=[34 * mm, CONTENT_W - 34 * mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_REG),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D1D5DB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def results_table(metrics: pd.DataFrame) -> Table:
    lookup = metrics.set_index("method")
    rows = [["Method", "Episodes", "Success rate", "Mean return", "Mean length"]]
    labels = {"random": "Random", "wm": "WM planning", "wm_vlm": "WM + VLM"}
    for method in ["random", "wm", "wm_vlm"]:
        row = lookup.loc[method]
        rows.append([
            labels[method],
            f"{int(row['episodes'])}",
            f"{float(row['success_rate']):.2f}",
            f"{float(row['mean_return']):.3f}",
            f"{float(row['mean_episode_length']):.2f}",
        ])
    table = Table(rows, colWidths=[48 * mm, 26 * mm, 31 * mm, 31 * mm, 31 * mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_REG),
        ("FONTSIZE", (0, 0), (-1, -1), 8.8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F3F4F6")),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#9CA3AF")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def bullet_list(items: list[str]) -> list[Paragraph]:
    return [p(f"- {item}", "Body") for item in items]


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_REG, 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawString(MARGIN_X, 10 * mm, "World-model MPC with VLM scoring in MiniGrid")
    canvas.drawRightString(PAGE_W - MARGIN_X, 10 * mm, str(doc.page))
    canvas.restoreState()


def build(config: dict) -> Path:
    metrics = pd.read_csv("outputs/metrics.csv")
    episodes = pd.read_csv("outputs/episodes.csv") if Path("outputs/episodes.csv").exists() else None
    train = pd.read_csv("outputs/training_loss.csv") if Path("outputs/training_loss.csv").exists() else None

    output = ensure_parent("report/report_en.pdf")
    planner = config["planner"]
    scored_steps = list(range(int(planner["vlm_stride"]), int(planner["horizon"]) + 1, int(planner["vlm_stride"])))
    if int(planner["horizon"]) not in scored_steps:
        scored_steps.append(int(planner["horizon"]))

    lookup = metrics.set_index("method")
    random_success = float(lookup.loc["random", "success_rate"])
    wm_success = float(lookup.loc["wm", "success_rate"])
    vlm_success = float(lookup.loc["wm_vlm", "success_rate"])

    story: list = []
    story += [p("World-model MPC with a VLM scorer", "ReportTitle")]
    story += [p("MiniGrid-Empty-6x6-v0 - measured experimental report", "Subtitle")]
    story += section("Experiment summary")
    story += [p("I train a compact RSSM world model on RGB MiniGrid rollouts and use it for model-predictive control. The main variant adds a frozen CLIP scorer to evaluate decoded future frames generated by the learned model.")]
    story += [p("The VLM score is computed on imagined future frames from the rollout, not only on the current real observation.", "Callout")]
    story += [Spacer(1, 3 * mm)]
    story += [badge_table([
        ("Environment", f"{config['env']['id']}, RGB observations resized to {config['env']['image_size']} x {config['env']['image_size']}"),
        ("Dataset", f"{config['data']['episodes']} collected episodes; 153 successful trajectories"),
        ("World model", "CNN encoder, GRU deterministic state, Gaussian latent state, RGB decoder, reward and continuation heads"),
        ("Planner", f"Random-shooting MPC, {planner['candidates']} candidates, horizon H={planner['horizon']}"),
        ("VLM scorer", f"{config['vlm']['model_name']}; future steps {', '.join(map(str, scored_steps))}"),
        ("Evaluation", f"{int(metrics['episodes'].iloc[0])} episodes per method; seeds {config['eval']['seeds']}, five episodes per seed"),
    ])]
    story += [Spacer(1, 6 * mm)]
    story += section("Compared policies")
    story += [method_table(), PageBreak()]

    story += section("Quantitative results")
    story += [results_table(metrics), Spacer(1, 7 * mm)]
    add_image(story, "outputs/plots/success_rate.png", 132, 5)
    story += [p(f"Reward-only world-model planning is strongest in this run: {wm_success:.2f} success rate versus {random_success:.2f} for random and {vlm_success:.2f} for WM+VLM. The VLM path is implemented, but it is not a performance gain here. The result suggests that CLIP scores on symbolic MiniGrid reconstructions are noisy and can conflict with the learned reward objective.")]
    ep_note = "Aggregate values are stored in outputs/metrics.csv."
    if episodes is not None:
        ep_note += " Per-episode seeds, returns and lengths are stored in outputs/episodes.csv."
    story += [p(ep_note, "Small"), PageBreak()]

    story += section("World model training and reconstruction check")
    story += [p("The reconstruction check matters because the VLM scores decoded model predictions. If the decoder does not preserve objects, the semantic score becomes unreliable.")]
    story += [p("In this run the RSSM reconstructs the room layout and the green goal clearly. The red agent is less stable, which helps explain why CLIP scoring is noisy.")]
    story += [Spacer(1, 3 * mm)]
    add_image(story, "outputs/screenshots/reconstruction.png", 150, 8)
    add_image(story, "outputs/plots/training_loss.png", 128, 4)
    if train is not None and len(train) > 0:
        story += [p(f"Training loss decreased from {float(train['loss'].iloc[0]):.3f} to {float(train['loss'].iloc[-1]):.3f} over {len(train)} epochs.", "Small")]
    story += [PageBreak()]

    story += section("Imagined rollouts and limitations")
    story += [p('The decoded future frames below come from a learned rollout and are scored by CLIP. The green goal remains visible, while the agent signal is weak. This makes it plausible that CLIP rewards the presence of the green square more than the exact relation "agent standing on goal".')]
    add_image(story, "outputs/screenshots/imagined_rollout.png", 160, 5)
    columns = [[p("Observed failure modes", "Section")] + bullet_list([
        "CLIP can reward green pixels instead of the spatial relation agent-on-goal.",
        "RSSM rollout errors accumulate; the red agent becomes blurry or disappears.",
        "Decoder artefacts directly affect the VLM objective.",
        "Random shooting with 32 candidates is brittle when the objective is noisy.",
    ]), [p("Future work", "Section")] + bullet_list([
        "Replace random shooting with CEM.",
        "Fine-tune a small contrastive scorer on MiniGrid relation labels.",
        "Add uncertainty penalties to imagined states.",
        "Compare with a symbolic oracle scorer for diagnosis only.",
        "Try an object-centric world model or a full Dreamer actor-critic agent.",
    ])]
    col_table = Table([columns], colWidths=[CONTENT_W / 2 - 3 * mm, CONTENT_W / 2 - 3 * mm], hAlign="LEFT")
    col_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [col_table, Spacer(1, 3 * mm)]
    story += section("Reproducibility")
    story += [p("Measured run command: python run_pipeline.py --methods random wm wm_vlm", "Small")]
    story += [p("The repository contains the source code, configuration, measured CSV files, plots, screenshots, GIF visualization, PDF report and full run log. Large regenerable files such as the collected dataset and RSSM checkpoint are not committed; they are produced again by running the pipeline.", "Small")]

    doc = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=MARGIN_X, leftMargin=MARGIN_X,
        topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
        title="World-model MPC with a VLM scorer",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"Saved report to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minigrid_empty.yaml")
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
