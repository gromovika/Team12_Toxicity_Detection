"""Reproduce cleaning and 70/15/15 holdout from Experiments_Data_Retraining 2.ipynb."""
from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ("neutral", "offensive", "toxic")
LABEL_MAP = {"NORMAL": "neutral", "INSULT": "offensive", "OBSCENITY": "offensive", "THREAT": "toxic"}
LINE_RE = re.compile(r"^__label__(\w+)\s+(.+)$")


def dataset_sha256(path: str | Path) -> str:
    """SHA-256 original UTF-8 dataset.txt bytes (same for .txt and .txt.zip).

    The baseline metadata hashes the uncompressed text file, not ZIP bytes.
    This function therefore uses the dataset member's bytes for ZIP inputs.
    """
    path = Path(path)
    digest = hashlib.sha256()
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith('.txt') and not n.startswith('__MACOSX/')]
            if len(names) != 1:
                raise ValueError(f'В ZIP ожидался ровно один TXT датасет, найдено: {len(names)}')
            with archive.open(names[0]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
    else:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def read_labeled_dataset(path: str | Path) -> tuple[pd.DataFrame, int]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Датасет не найден: {path}")
    rows, malformed = [], 0  # malformed includes valid multi-label lines skipped by original trainer
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        names = [n for n in archive.namelist() if n.endswith('.txt') and not n.startswith('__MACOSX/')]
        if not names:
            archive.close()
            raise ValueError('ZIP без .txt датасета')
        stream = archive.open(names[0])
    else:
        archive = None
        stream = path.open('rb')
    try:
        for line in stream:
            match = LINE_RE.match(line.decode('utf-8', errors='replace').strip())
            if match is None or match.group(1) not in LABEL_MAP:
                malformed += 1
                continue
            rows.append((match.group(2), match.group(1), LABEL_MAP[match.group(1)]))
    finally:
        stream.close()
        if archive is not None: archive.close()
    if not rows:
        raise ValueError('Нет размеченных строк')
    return pd.DataFrame(rows, columns=['text','source_label','label']), malformed


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text).lower().replace('ё', 'е')).strip()


def split_like_original(data: pd.DataFrame, seed: int = 42):
    """Exactly match original notebook preprocessing and split order.

    Never use a different seed or algorithm when recomputing the baseline holdout.
    """
    raw = data.copy()
    raw['text_clean'] = raw.text.map(normalize_text)
    raw = raw[raw.text_clean.str.len() > 0].copy()
    conflicts = raw.groupby('text_clean').label.nunique()
    bad_texts = set(conflicts[conflicts > 1].index)
    clean = raw[~raw.text_clean.isin(bad_texts)].drop_duplicates('text_clean').copy()
    if set(clean.label.unique()) != set(CLASSES):
        raise ValueError('Для стратификации нужны все три класса')
    train_val, test = train_test_split(
        clean[['text_clean','label']], test_size=.15, random_state=seed,
        stratify=clean.label,
    )
    train, val = train_test_split(
        train_val, test_size=.15/.85, random_state=seed,
        stratify=train_val.label,
    )
    assert not (set(train.text_clean) & set(test.text_clean))
    assert not (set(val.text_clean) & set(test.text_clean))
    assert not (set(train.text_clean) & set(val.text_clean))
    return train, val, test
