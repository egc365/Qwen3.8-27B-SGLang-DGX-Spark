#!/usr/bin/env python3

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from start_classifier import profile_fingerprint


PREFLIGHT = '''import json, subprocess, sys
profile = json.loads(sys.argv[1])
names = subprocess.check_output(["docker", "ps", "-a", "--format", "{{.Names}}"], text=True).splitlines()
if profile["container"] in names:
    prior = json.loads(subprocess.check_output(["docker", "inspect", profile["container"]], text=True))[0]
    if (prior["Config"].get("Labels") or {}).get("q27.profile") != profile["fingerprint"]:
        sys.exit("Classifier profile differs; preserve existing service before any file transfer")
    if not prior["State"]["Running"]:
        sys.exit("Classifier is stopped; preserve existing service and inspect logs before any file transfer")
'''


def deployment_plan(root, target_host=None):
    root = Path(root).resolve()
    config = root / "classifier.json"
    cfg = json.loads(config.read_text())
    directory = cfg["remote_directory"]
    if not isinstance(directory, str) or not re.fullmatch(r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", directory):
        raise ValueError("remote_directory must be a simple relative path without shell characters or '..'")
    host = target_host or cfg["host"]
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host):
        raise ValueError("target host must be a hostname or IPv4 address")
    image = cfg["image"]
    if not isinstance(image, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
        raise ValueError("image must be pinned by sha256")
    source = Path(cfg["source_checkpoint"])
    if not source.is_absolute() or not (source / "config.json").is_file():
        raise ValueError(f"existing checkpoint missing at {source}; this command does not download weights")
    runtime = root / ".scratch/runtime"
    if not (runtime / "sglang/__init__.py").is_file() or not (runtime / "overlay-receipt.json").is_file():
        raise ValueError("prepared full .scratch/runtime and overlay-receipt.json required; "
                         "run python3 tools/prepare_classifier.py before deployment")
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host]
    remote = f'"$HOME/{directory}"'
    rsync = ["rsync", "-a", "--"]
    profile = {"container": cfg["container"],
               "fingerprint": profile_fingerprint(cfg, (runtime / "overlay-receipt.json").read_bytes())}
    return [
        {"name": "preflight", "command": ssh + [shlex.join(["python3", "-c", PREFLIGHT, json.dumps(profile)])]},
        {"name": "create-directories", "command": ssh + ["mkdir -p " + " ".join(f'"$HOME/{directory}/{part}"' for part in ("tools", "models/original-nvfp4", "runtime", "state"))]},
        {"name": "checkpoint", "command": rsync + [str(source) + "/", f"{host}:{directory}/models/original-nvfp4/"]},
        {"name": "pinned-image", "inspect": ssh + ["docker image inspect " + shlex.quote(image)],
         "save": ["docker", "image", "save", image], "load": ssh + ["docker image load"]},
        {"name": "runtime", "command": rsync + [str(runtime) + "/", f"{host}:{directory}/runtime/"]},
        {"name": "tools", "command": rsync + [str(root / "tools") + "/", f"{host}:{directory}/tools/"]},
        {"name": "config", "command": rsync + [str(config), f"{host}:{directory}/classifier.json"]},
        {"name": "start", "command": ssh + [f"python3 {remote}/tools/start_classifier.py --root {remote}"]},
    ]


def transfer_image(save_command, load_command):
    save = subprocess.Popen(save_command, stdout=subprocess.PIPE)
    try:
        load = subprocess.Popen(load_command, stdin=save.stdout)
    except BaseException:
        save.stdout.close()
        save.terminate()
        save.wait()
        raise
    save.stdout.close()
    load_code = load.wait()
    save_code = save.wait()
    if save_code or load_code:
        raise RuntimeError(f"image transfer failed: docker save exit {save_code}; remote load exit {load_code}")


def apply_plan(plan):
    for step in plan:
        if "command" in step:
            subprocess.run(step["command"], check=True)
        else:
            inspected = subprocess.run(step["inspect"], stdout=subprocess.DEVNULL, check=False)
            if inspected.returncode == 1:
                transfer_image(step["save"], step["load"])
            elif inspected.returncode != 0:
                raise RuntimeError(f"remote image inspection failed with exit {inspected.returncode}")


def main():
    parser = argparse.ArgumentParser(description='Transfer a prepared classifier to its configured host and start it once.')
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target-host", help="transfer address override; saved client routing stays unchanged")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        plan = deployment_plan(args.root, args.target_host)
        if args.plan:
            print(json.dumps(plan, indent=2))
        else:
            apply_plan(plan)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
