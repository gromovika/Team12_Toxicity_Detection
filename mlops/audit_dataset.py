"""Summarize original dataset without storing texts or PII in output."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import zipfile

from mlops.data import CLASSES, dataset_sha256, normalize_text, read_labeled_dataset, split_like_original

HEAD_RE = re.compile(r"^((?:__label__[A-Z]+,)*__label__[A-Z]+)\s+")


def audit(path: str | Path) -> dict:
    path = Path(path)
    groups = Counter()
    with zipfile.ZipFile(path) if path.suffix.lower() == '.zip' else open(path, 'rb') as source:
        if isinstance(source, zipfile.ZipFile):
            names = [n for n in source.namelist() if n.endswith('.txt') and not n.startswith('__MACOSX/')]
            if len(names) != 1:
                raise ValueError('ZIP должен содержать ровно один TXT датасет')
            handle = source.open(names[0])
        else:
            handle = source
        for line in handle:
            m = HEAD_RE.match(line.decode('utf-8', errors='replace'))
            groups[m.group(1) if m else '<UNRECOGNIZED>'] += 1
        if handle is not source:
            handle.close()
    raw, skipped = read_labeled_dataset(path)
    train, valid, test = split_like_original(raw)
    return {
        'dataset_sha256': dataset_sha256(path),
        'total_source_lines': int(sum(groups.values())),
        'source_label_combinations': {k.replace('__label__',''):v for k,v in sorted(groups.items())},
        'included_single_label_lines': len(raw),
        'skipped_multilabel_or_invalid_lines': skipped,
        'original_class_counts': {k:int((raw.label==k).sum()) for k in CLASSES},
        'clean_unique_records': len(train)+len(valid)+len(test),
        'removed_during_normalization_or_deduplication': len(raw)-len(train)-len(valid)-len(test),
        'split': {'train':len(train),'validation':len(valid),'test':len(test)},
        'test_class_counts': {k:int((test.label==k).sum()) for k in CLASSES},
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/dataset.txt.zip')
    parser.add_argument('--output',default='reports/dataset_audit.json')
    args=parser.parse_args()
    result=audit(args.data)
    out=Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
