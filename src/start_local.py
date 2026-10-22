"""Start the portable LLM service in the background, bound to loopback only."""
import os
import subprocess
import time
from pathlib import Path
import requests
from src.setup_local import MODEL_FILE, ROOT

def main():
    try:
        health = requests.get('http://127.0.0.1:8080/health', timeout=2)
        if health.status_code == 200:
            models = requests.get('http://127.0.0.1:8080/v1/models', timeout=2).json()
            if 'wine-local' not in {model['id'] for model in models.get('data', [])}:
                raise SystemExit('Port 8080 belongs to another model service. Stop it before starting the wine model.')
            print('The local wine model service is already running at port 8080.')
            return
    except requests.RequestException:
        pass
    servers = list((ROOT / 'local_runtime').rglob('llama-server.exe'))
    if len(servers) != 1 or not (ROOT / 'local_llm' / MODEL_FILE).exists():
        raise SystemExit('First run: python -m src.setup_local')
    logs = ROOT / 'local_runtime'
    command = [str(servers[0]), '--model', str(ROOT / 'local_llm' / MODEL_FILE),
               '--host', '127.0.0.1', '--port', '8080', '--ctx-size', '4096',
               '--n-gpu-layers', '99', '--device', 'Vulkan0', '--alias', 'wine-local']
    with (logs / 'stdout.log').open('a') as stdout, (logs / 'stderr.log').open('a') as stderr:
        process = subprocess.Popen(command, cwd=str(servers[0].parent),
            stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (logs / 'server.pid').write_text(str(process.pid))
    print(f'Started local model process {process.pid}. Waiting for the model to load...', flush=True)
    for _ in range(240):
        if process.poll() is not None:
            raise SystemExit('The model server stopped. Inspect local_runtime/stderr.log.')
        try:
            if requests.get('http://127.0.0.1:8080/health', timeout=2).status_code == 200:
                print('Local Qwen model is ready; no paid API is used.', flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise SystemExit('Model is still loading. Inspect local_runtime/stderr.log before retrying.')

if __name__ == '__main__':
    main()
