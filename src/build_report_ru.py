from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image as PILImage, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path('.')
OUT = Path('report/report_ru.pdf')
import tempfile
ASSETS = Path(tempfile.mkdtemp(prefix='report_ru_assets_'))

# Fonts for Cyrillic text.
FONT_REG = 'DejaVuSans'
FONT_BOLD = 'DejaVuSans-Bold'
pdfmetrics.registerFont(TTFont(FONT_REG, '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont(FONT_BOLD, '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))

PAGE_W, PAGE_H = A4
MARGIN_X = 18 * mm
MARGIN_TOP = 17 * mm
MARGIN_BOTTOM = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name='ReportTitle', parent=styles['Title'], fontName=FONT_BOLD, fontSize=21,
    leading=26, alignment=TA_LEFT, textColor=colors.HexColor('#111827'), spaceAfter=7,
))
styles.add(ParagraphStyle(
    name='Subtitle', parent=styles['Normal'], fontName=FONT_REG, fontSize=10.2,
    leading=13.2, textColor=colors.HexColor('#4B5563'), spaceAfter=12,
))
styles.add(ParagraphStyle(
    name='Section', parent=styles['Heading2'], fontName=FONT_BOLD, fontSize=14.5,
    leading=18, textColor=colors.HexColor('#111827'), spaceBefore=7, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name='Body', parent=styles['BodyText'], fontName=FONT_REG, fontSize=9.5,
    leading=12.9, textColor=colors.HexColor('#111827'), spaceAfter=5,
))
styles.add(ParagraphStyle(
    name='Small', parent=styles['BodyText'], fontName=FONT_REG, fontSize=8.2,
    leading=10.7, textColor=colors.HexColor('#374151'), spaceAfter=4,
))
styles.add(ParagraphStyle(
    name='Callout', parent=styles['BodyText'], fontName=FONT_BOLD, fontSize=9.9,
    leading=13.2, textColor=colors.HexColor('#111827'), spaceAfter=5,
))
styles.add(ParagraphStyle(
    name='Cell', parent=styles['BodyText'], fontName=FONT_REG, fontSize=8.5,
    leading=10.6, textColor=colors.HexColor('#111827'), spaceAfter=0,
))
styles.add(ParagraphStyle(
    name='CellBold', parent=styles['BodyText'], fontName=FONT_BOLD, fontSize=8.5,
    leading=10.6, textColor=colors.HexColor('#111827'), spaceAfter=0,
))


def p(text: str, style: str = 'Body') -> Paragraph:
    return Paragraph(text, styles[style])


def section(title: str):
    return [
        Paragraph(title, styles['Section']),
        HRFlowable(width='100%', thickness=0.7, color=colors.HexColor('#E5E7EB'), spaceBefore=0, spaceAfter=8),
    ]


def add_image(story: list, path: str | Path, width_mm: float, space_after_mm: float = 4):
    path = Path(path)
    img = Image(str(path))
    w, h = img.imageWidth, img.imageHeight
    img.drawWidth = width_mm * mm
    img.drawHeight = img.drawWidth * h / w
    story.append(img)
    if space_after_mm:
        story.append(Spacer(1, space_after_mm * mm))


def table_style(font_size: float = 8.3) -> TableStyle:
    return TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
        ('FONTNAME', (0, 1), (-1, -1), FONT_REG),
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3F4F6')),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#D1D5DB')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ])


def badge_table(items: Iterable[tuple[str, str]]) -> Table:
    rows = [[p(label, 'CellBold'), p(value, 'Cell')] for label, value in items]
    table = Table(rows, colWidths=[38 * mm, CONTENT_W - 38 * mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F9FAFB')),
        ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#D1D5DB')),
        ('INNERGRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#E5E7EB')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    return table


def methods_table() -> Table:
    rows = [
        [p('Метод', 'CellBold'), p('Целевая функция планировщика', 'CellBold')],
        [p('random', 'Cell'), p('Случайные действия.', 'Cell')],
        [p('wm', 'Cell'), p('Планирование по модели мира: сумма предсказанных вознаграждений.', 'Cell')],
        [p('wm_vlm', 'Cell'), p('Тот же планировщик плюс максимальная CLIP-оценка по будущим декодированным кадрам.', 'Cell')],
    ]
    table = Table(rows, colWidths=[32 * mm, CONTENT_W - 32 * mm])
    table.setStyle(table_style(8.4))
    return table


def results_table(metrics: pd.DataFrame) -> Table:
    lookup = metrics.set_index('method')
    labels = {
        'random': 'Случайный',
        'wm': 'Модель мира',
        'wm_vlm': 'Модель мира + CLIP',
    }
    rows = [[
        p('Метод', 'CellBold'),
        p('Эпизоды', 'CellBold'),
        p('Доля успеха', 'CellBold'),
        p('Среднее вознаграждение', 'CellBold'),
        p('Средняя длина', 'CellBold'),
    ]]
    for method in ['random', 'wm', 'wm_vlm']:
        row = lookup.loc[method]
        rows.append([
            p(labels[method], 'Cell'),
            p(str(int(row['episodes'])), 'Cell'),
            p(f"{float(row['success_rate']):.2f}", 'Cell'),
            p(f"{float(row['mean_return']):.3f}", 'Cell'),
            p(f"{float(row['mean_episode_length']):.2f}", 'Cell'),
        ])
    table = Table(rows, colWidths=[42 * mm, 23 * mm, 29 * mm, 42 * mm, 31 * mm])
    table.setStyle(table_style(8.1))
    table.setStyle(TableStyle([
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return table


def bullet_list(items: list[str]) -> list[Paragraph]:
    return [p(f'- {item}', 'Body') for item in items]


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_REG, 8)
    canvas.setFillColor(colors.HexColor('#6B7280'))
    canvas.drawString(MARGIN_X, 10 * mm, 'Планирование по модели мира в MiniGrid')
    canvas.drawRightString(PAGE_W - MARGIN_X, 10 * mm, str(doc.page))
    canvas.restoreState()


def load_font(size: int, bold: bool = False):
    path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    return ImageFont.truetype(path, size)


def make_success_chart(metrics: pd.DataFrame) -> Path:
    out = ASSETS / 'success_rate_ru.png'
    lookup = metrics.set_index('method')
    names = ['Случайный', 'Модель мира', 'Модель мира + CLIP']
    values = [lookup.loc[m, 'success_rate'] for m in ['random', 'wm', 'wm_vlm']]
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.bar(names, values, color=['#8A8A8A', '#4C78A8', '#59A14F'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Доля успешных эпизодов')
    ax.set_title('Сравнение методов в MiniGrid')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def make_training_chart(train: pd.DataFrame) -> Path:
    out = ASSETS / 'training_loss_ru.png'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(6.0, 3.35))
    ax.plot(train['epoch'], train['loss'], marker='o', linewidth=1.5)
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Ошибка обучения')
    ax.set_title('Обучение модели мира')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def make_reconstruction_figure() -> Path:
    src = PILImage.open(ROOT / 'outputs/screenshots/reconstruction.png').convert('RGB')
    # Crop the English titles from the original figure and rebuild captions in Russian.
    w, h = src.size
    left = src.crop((39, 42, 454, 455))
    right = src.crop((507, 42, 921, 455))
    out = PILImage.new('RGB', (700, 345), 'white')
    draw = ImageDraw.Draw(out)
    title_font = load_font(20, bold=True)
    draw.text((90, 6), 'Реальный кадр', fill=(17, 24, 39), font=title_font)
    draw.text((415, 6), 'Реконструкция RSSM', fill=(17, 24, 39), font=title_font)
    out.paste(left.resize((285, 285), PILImage.Resampling.NEAREST), (28, 45))
    out.paste(right.resize((285, 285), PILImage.Resampling.NEAREST), (382, 45))
    path = ASSETS / 'reconstruction_ru.png'
    out.save(path)
    return path


def make_rollout_figure() -> Path:
    src = PILImage.open(ROOT / 'outputs/screenshots/imagined_rollout.png').convert('RGB')
    # Original image contains English labels. Crop frames and recreate captions in Russian.
    crops = [
        (87, 128, 444, 486, '+3', '0.398'),
        (592, 128, 950, 486, '+6', '0.448'),
        (1098, 128, 1455, 486, '+9', '0.372'),
        (1604, 128, 1960, 486, '+12', '0.347'),
    ]
    frame_size = 118
    gap = 32
    top = 44
    out_w = 4 * frame_size + 3 * gap + 40
    out_h = 195
    out = PILImage.new('RGB', (out_w, out_h), 'white')
    draw = ImageDraw.Draw(out)
    title_font = load_font(15, bold=True)
    cap_font = load_font(13)
    draw.text((20, 5), 'Будущие кадры из воображаемой развёртки, оценённые CLIP', fill=(17, 24, 39), font=title_font)
    x = 20
    for x0, y0, x1, y1, step, score in crops:
        frame = src.crop((x0, y0, x1, y1)).resize((frame_size, frame_size), PILImage.Resampling.NEAREST)
        caption = f'шаг {step}\nCLIP={score}'
        draw.multiline_text((x + 22, 24), caption, fill=(17, 24, 39), font=cap_font, align='center', spacing=1)
        out.paste(frame, (x, top + 22))
        x += frame_size + gap
    path = ASSETS / 'imagined_rollout_ru.png'
    out.save(path)
    return path


def build_pdf() -> Path:
    metrics = pd.read_csv(ROOT / 'outputs/metrics.csv')
    train = pd.read_csv(ROOT / 'outputs/training_loss.csv')
    success_chart = make_success_chart(metrics)
    training_chart = make_training_chart(train)
    reconstruction_fig = make_reconstruction_figure()
    rollout_fig = make_rollout_figure()

    lookup = metrics.set_index('method')
    random_success = float(lookup.loc['random', 'success_rate'])
    wm_success = float(lookup.loc['wm', 'success_rate'])
    vlm_success = float(lookup.loc['wm_vlm', 'success_rate'])

    story: list = []
    story += [p('Планирование по модели мира со зрительно-языковой оценкой', 'ReportTitle')]
    story += [p('MiniGrid-Empty-6x6-v0 - отчёт по измеренному запуску', 'Subtitle')]
    story += section('Краткое описание')
    story += [p('В работе обучена компактная модель мира RSSM для RGB-наблюдений MiniGrid. Затем эта модель используется для планирования: агент строит воображаемые будущие состояния, декодирует их в кадры и выбирает действие по оценке нескольких кандидатов.')]
    story += [p('Отдельный вариант планировщика добавляет CLIP-оценку будущих кадров. Эта оценка считается именно на кадрах из будущей развёртки модели, а не только на текущем наблюдении.', 'Callout')]
    story += [Spacer(1, 3 * mm)]
    story += [badge_table([
        ('Среда', 'MiniGrid-Empty-6x6-v0; RGB-наблюдения приведены к 64 x 64'),
        ('Данные', '240 собранных эпизодов; 153 успешные траектории'),
        ('Модель мира', 'Свёрточный кодировщик, GRU-состояние, гауссова скрытая переменная, декодер кадра, головы вознаграждения и продолжения'),
        ('Планирование', 'Случайный перебор последовательностей действий: 32 кандидата, горизонт H=12'),
        ('Оцениватель', 'openai/clip-vit-base-patch32; оцениваются будущие шаги 3, 6, 9, 12'),
        ('Оценка качества', '25 эпизодов на метод; начальные значения [0, 1, 2, 3, 4], по пять эпизодов на значение'),
    ])]
    story += [Spacer(1, 6 * mm)]
    story += section('Сравниваемые методы')
    story += [methods_table(), PageBreak()]

    story += section('Количественные результаты')
    story += [results_table(metrics), Spacer(1, 6 * mm)]
    add_image(story, success_chart, 132, 5)
    story += [p(f'Лучший результат в этом запуске дал планировщик по модели мира без CLIP-оценки: доля успеха {wm_success:.2f}. Случайная стратегия дала {random_success:.2f}, а вариант с CLIP-оценкой - {vlm_success:.2f}.')]
    story += [p('Добавление CLIP-оценки ухудшило результат. Это не ошибка протокола: оцениватель действительно применяется к будущим кадрам. Проблема в том, что готовая CLIP-модель плохо согласуется с пиксельными реконструкциями MiniGrid.', 'Body')]
    story += [p('Сводные значения лежат в outputs/metrics.csv. Данные по отдельным эпизодам лежат в outputs/episodes.csv.', 'Small'), PageBreak()]

    story += section('Обучение модели мира')
    story += [p('Проверка реконструкции важна, потому что CLIP оценивает не реальные кадры среды, а декодированные предсказания модели. Если декодер теряет объекты, текстово-визуальная оценка становится ненадёжной.')]
    story += [p('В этом запуске модель мира хорошо сохраняет структуру комнаты и зелёную цель. Красный агент восстанавливается хуже и становится слабым или размытым; это объясняет шум в CLIP-оценках.')]
    story += [Spacer(1, 3 * mm)]
    add_image(story, reconstruction_fig, 150, 7)
    add_image(story, training_chart, 128, 4)
    story += [p(f"Ошибка обучения снизилась с {float(train['loss'].iloc[0]):.3f} до {float(train['loss'].iloc[-1]):.3f} за {len(train)} эпох.", 'Small'), PageBreak()]

    story += section('Воображаемые развёртки и ограничения')
    story += [p('Ниже показаны будущие кадры, полученные развёрткой обученной модели мира. Зелёная цель остаётся хорошо видимой, а агент выражен слабо. Поэтому CLIP может реагировать на наличие зелёной клетки, а не на отношение "агент стоит на цели".')]
    add_image(story, rollout_fig, 160, 5)
    columns = [
        [p('Наблюдавшиеся сбои', 'Section')] + bullet_list([
            'CLIP иногда поощряет саму зелёную клетку вместо отношения агент-на-цели.',
            'Ошибки модели мира накапливаются по горизонту; красный агент размывается или исчезает.',
            'Артефакты декодера напрямую попадают в целевую функцию варианта с CLIP-оценкой.',
            'Случайный перебор 32 кандидатов нестабилен при шумной оценке.',
        ]),
        [p('Что улучшить дальше', 'Section')] + bullet_list([
            'Заменить случайный перебор на метод кросс-энтропии (CEM).',
            'Дообучить небольшой контрастивный оцениватель на отношениях в MiniGrid.',
            'Добавить штраф за неопределённые воображаемые состояния.',
            'Для диагностики сравнить CLIP с символическим оценивателем.',
            'Попробовать объектную модель мира или полный вариант Dreamer с актором-критиком.',
        ]),
    ]
    col_table = Table([columns], colWidths=[CONTENT_W / 2 - 3 * mm, CONTENT_W / 2 - 3 * mm])
    col_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 6)]))
    story += [col_table]

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title='Планирование по модели мира со зрительно-языковой оценкой',
        author='',
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return OUT


if __name__ == '__main__':
    print(build_pdf())
