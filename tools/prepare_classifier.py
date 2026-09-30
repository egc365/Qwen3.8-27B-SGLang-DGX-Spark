#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import uuid
from build_decision_overlay import build_overlay


def run(*args):
    subprocess.run(args, check=True)


def prepare(root):
    cfg = json.loads((root / 'classifier.json').read_text())
    run('docker', 'image', 'inspect', '--format', '{{.Id}}', cfg['image'])
    scratch = root / '.scratch'
    scratch.mkdir(exist_ok=True)
    upstream = scratch / 'native-upstream'
    if not upstream.exists():
        run('git', 'init', str(upstream))
        run('git', '-C', str(upstream), 'remote', 'add', 'origin', 'https://github.com/sgl-project/sglang.git')
    commit = cfg['upstream_commit']
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('upstream_commit must be a full hexadecimal commit hash')
    run('git', '-C', str(upstream), 'fetch', '--depth', '1', 'origin', commit)
    run('git', '-C', str(upstream), 'checkout', '--detach', commit)
    stage = scratch / ('runtime-stage-' + uuid.uuid4().hex[:12])
    stage.mkdir()
    base = stage / 'sglang'
    container = subprocess.check_output(['docker', 'create', '--entrypoint', 'true', cfg['image']], text=True).strip()
    try:
        run('docker', 'cp', container + ':/sgl-workspace/sglang/python/sglang', str(base))
    finally:
        run('docker', 'rm', container)
    overlay = stage / 'overlay'
    manifest = build_overlay(base, upstream / 'python/sglang', overlay)
    shutil.copytree(overlay / 'sglang', base, dirs_exist_ok=True)
    shutil.copy2(overlay / 'manifest.json', stage / 'overlay-receipt.json')
    shutil.rmtree(overlay)
    runtime = scratch / 'runtime'
    if runtime.exists():
        runtime.rename(scratch / ('runtime-previous-' + uuid.uuid4().hex[:12]))
    stage.rename(runtime)
    print(json.dumps({'runtime':str(runtime),'upstream_commit':commit,'image':cfg['image'],'patched_files':len(manifest['output_sha256'])}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare a pinned native API overlay; model weights and GPU services are untouched.')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    prepare(args.root.resolve())
