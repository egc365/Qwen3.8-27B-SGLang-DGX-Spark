#!/usr/bin/env python3
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def profile_fingerprint(cfg, manifest_bytes):
    launch_config = {k: v for k, v in cfg.items() if k not in ('source_checkpoint', 'remote_directory', 'upstream_commit')}
    return hashlib.sha256((json.dumps(launch_config, indent=2) + '\n').encode() + manifest_bytes).hexdigest()


def start(root, wait):
    config_bytes = (root / 'classifier.json').read_bytes()
    cfg = json.loads(config_bytes)
    if not (root / cfg['checkpoint'] / 'config.json').is_file():
        raise RuntimeError('Checkpoint is missing')
    if not (root / cfg['runtime'] / 'srt/entrypoints/openai/serving_decisions.py').is_file():
        raise RuntimeError('Native decision overlay is missing')
    state = root / 'state'
    state.mkdir(exist_ok=True)
    fingerprint = profile_fingerprint(cfg, (root / 'runtime/overlay-receipt.json').read_bytes())
    lock = (state / 'deployment.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    names = run('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
    existing = cfg['container'] in names
    if existing:
        prior = json.loads(run('docker', 'inspect', cfg['container']))[0]
        if prior['Config']['Labels'].get('q27.profile') != fingerprint:
            raise RuntimeError('Existing classifier has different configuration; preserve it and inspect before replacement')
        if not prior['State']['Running']:
            raise RuntimeError('Existing classifier stopped; inspect logs before deliberate restart')
    else:
        available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) / 1024**2
        if available < cfg['min_mem_available_gib']:
            raise RuntimeError(f'Only {available:.1f} GiB available; requires {cfg["min_mem_available_gib"]}')
        run('docker', 'image', 'inspect', cfg['image'])
        command = ['docker', 'run', '-d', '--name', cfg['container'], '--label', f'q27.profile={fingerprint}', '--ipc', 'host', '--device', 'nvidia.com/gpu=all', '-p', f'{cfg["port"]}:{cfg["port"]}', '-e', 'HF_HUB_OFFLINE=1', '-e', 'PYTHONPATH=/decision-runtime', '-v', f'{root / "models"}:/classifier-models:ro', '-v', f'{root / "runtime"}:/decision-runtime:ro', '-v', f'{state}:/classifier-state', '-e', 'TRITON_CACHE_DIR=/classifier-state/triton', cfg['image'], 'python3', '-m', 'sglang.launch_server', '--model-path', str(Path('/classifier-models') / Path(cfg['checkpoint']).relative_to('models')), '--served-model-name', cfg['model'], '--host', '0.0.0.0', '--port', str(cfg['port']), '--trust-remote-code', '--kv-cache-dtype', 'bfloat16', '--attention-backend', 'flashinfer', '--mem-fraction-static', str(cfg['mem_fraction_static']), '--context-length', str(cfg['context_length']), '--max-total-tokens', str(cfg['max_total_tokens']), '--max-running-requests', str(cfg['max_running_requests']), '--max-mamba-cache-size', str(cfg['max_mamba_cache_size']), '--mamba-ssm-dtype', 'bfloat16', '--mamba-radix-cache-strategy', 'extra_buffer', '--chunked-prefill-size', '2048', '--reasoning-parser', 'qwen3', '--disable-cuda-graph']
        print(json.dumps({'container_id':run(*command),'before_mem_available_gib':available,'fingerprint':fingerprint}), flush=True)
    started = time.monotonic()
    while time.monotonic() - started < wait:
        try:
            with urlopen(f'http://127.0.0.1:{cfg["port"]}/health', timeout=2) as response:
                if response.status == 200:
                    print(json.dumps({'ready':True,'readiness_wait_seconds':time.monotonic()-started,'url':f'http://{cfg["host"]}:{cfg["port"]}/v1','existing':existing}), flush=True)
                    return
        except (OSError, ValueError):
            pass
        if run('docker', 'inspect', '--format', '{{.State.Running}}', cfg['container']) != 'true':
            raise RuntimeError('Classifier exited; inspect docker logs qwen27-classifier')
        time.sleep(3)
    raise RuntimeError('Readiness deadline expired; inspect classifier logs before any restart')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--wait', type=int, default=900)
    args = parser.parse_args()
    start(args.root.resolve(), args.wait)
