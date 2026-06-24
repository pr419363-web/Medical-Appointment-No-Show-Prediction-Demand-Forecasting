"""
Collect models and artifacts into a release/ folder for sharing.
Usage:
    python scripts/collect_models.py
"""
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / 'models'
NOTEBOOKS = BASE / 'notebooks'
DOCS = BASE / 'docs'
RELEASE = BASE / 'release'

RELEASE.mkdir(exist_ok=True)
(RELEASE / 'models').mkdir(exist_ok=True)
(RELEASE / 'notebooks').mkdir(exist_ok=True)
(RELEASE / 'docs').mkdir(exist_ok=True)

# copy models
if MODELS.exists():
    for f in MODELS.iterdir():
        if f.is_file():
            shutil.copy2(f, RELEASE / 'models' / f.name)

# copy notebooks
if NOTEBOOKS.exists():
    for f in NOTEBOOKS.glob('*.ipynb'):
        shutil.copy2(f, RELEASE / 'notebooks' / f.name)

# copy docs
if DOCS.exists():
    for f in DOCS.iterdir():
        if f.is_file():
            shutil.copy2(f, RELEASE / 'docs' / f.name)

print('Release bundle prepared at', RELEASE)
print('Include `data/processed_medical_appointments.csv` manually if you want reproducibility.')
