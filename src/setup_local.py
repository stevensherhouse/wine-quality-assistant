"""Download a pinned portable Windows runtime and public GGUF model."""
import hashlib
import json
import zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_TAG = 'b11669'
RUNTIME_NAME = f'llama-{RUNTIME_TAG}-bin-win-vulkan-x64.zip'
RUNTIME_URL = f'https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_TAG}/{RUNTIME_NAME}'
RUNTIME_SHA256 = '2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e'
MODEL_REPO = 'bartowski/Qwen2.5-7B-Instruct-GGUF'
MODEL_FILE = 'Qwen2.5-7B-Instruct-Q4_K_M.gguf'

def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def download(url, path, expected):
    if path.exists() and file_hash(path) == expected:
        print(f'Already verified: {path.name}', flush=True)
        return
    temporary = path.with_suffix(path.suffix + '.part')
    print(f'Downloading {path.name}', flush=True)
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        size = 0
        last_progress = 0
        digest = hashlib.sha256()
        with temporary.open('wb') as destination:
            for chunk in response.iter_content(8 * 1024 * 1024):
                destination.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if size - last_progress >= 256 * 1024 * 1024:
                    print(f'  {size / 2**30:.2f} GiB downloaded', flush=True)
                    last_progress = size
    if digest.hexdigest() != expected:
        raise ValueError(f'Checksum mismatch for {path.name}; partial download retained for inspection')
    temporary.replace(path)
    print(f'Checksum verified: {path.name}', flush=True)

def main():
    runtime = ROOT / 'local_runtime'
    model_dir = ROOT / 'local_llm'
    runtime.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    download(RUNTIME_URL, runtime / RUNTIME_NAME, RUNTIME_SHA256)
    with zipfile.ZipFile(runtime / RUNTIME_NAME) as archive:
        for member in archive.infolist():
            target = (runtime / member.filename).resolve()
            if not target.is_relative_to(runtime.resolve()):
                raise ValueError('Invalid archive member path')
        archive.extractall(runtime)
    response = requests.get(f'https://huggingface.co/api/models/{MODEL_REPO}?blobs=true', timeout=30)
    response.raise_for_status()
    info = response.json()
    model = next(item for item in info['siblings'] if item['rfilename'] == MODEL_FILE)
    sha256 = model['lfs']['sha256']
    revision = info['sha']
    url = f'https://huggingface.co/{MODEL_REPO}/resolve/{revision}/{MODEL_FILE}'
    download(url, model_dir / MODEL_FILE, sha256)
    manifest = {'runtime_tag': RUNTIME_TAG, 'runtime_sha256': RUNTIME_SHA256,
                'model_repo': MODEL_REPO, 'model_revision': revision,
                'model_file': MODEL_FILE, 'model_sha256': sha256,
                'upstream': 'Qwen/Qwen2.5-7B-Instruct'}
    (model_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (ROOT / 'reports/local-model.json').write_text(json.dumps(manifest, indent=2))
    print('Local runtime and model are ready. Run: python -m src.start_local', flush=True)

if __name__ == '__main__':
    main()
