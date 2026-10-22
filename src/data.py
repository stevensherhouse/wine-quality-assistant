"""Download the original UCI files; record their exact content hashes."""
import hashlib
import io
import json
import zipfile
from pathlib import Path
import pandas as pd
import requests
from src.preprocess import clean_frame

URL = 'https://archive.ics.uci.edu/static/public/186/wine+quality.zip'

def download_data(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    response = requests.get(URL, timeout=90)
    response.raise_for_status()
    hashes = {}
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for kind in ['red', 'white']:
            name = f'winequality-{kind}.csv'
            candidates = [n for n in archive.namelist() if n.endswith(name)]
            if len(candidates) != 1:
                raise ValueError(f'Expected one {name} in the UCI archive')
            content = archive.read(candidates[0])
            pd.read_csv(io.BytesIO(content), sep=';')[['quality']]
            (directory / name).write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
    manifest = {'source': URL, 'sha256': hashes}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest

def load_data(directory: Path) -> tuple[pd.DataFrame, dict]:
    frames, hashes = [], {}
    for kind in ['red', 'white']:
        path = directory / f'winequality-{kind}.csv'
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path, sep=';')
        frame['wine_type'] = kind
        frames.append(frame)
    result = clean_frame(pd.concat(frames, ignore_index=True))
    if not result.quality.between(0, 10).all():
        raise ValueError('Quality target must be between 0 and 10')
    # Prevent identical feature vectors, including conflicting labels, crossing splits.
    from src.preprocess import FEATURES
    result = result.groupby(FEATURES, dropna=False, as_index=False)['quality'].mean()
    return result, {'source': URL, 'sha256': hashes, 'rows_after_grouping': len(result)}
