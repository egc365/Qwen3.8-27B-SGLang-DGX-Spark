import contextlib
import io
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from deploy_classifier import apply_plan, deployment_plan, main, transfer_image
from start_classifier import profile_fingerprint


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / ".scratch"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        source.mkdir()
        (source / "config.json").write_text("{}")
        runtime = self.root / ".scratch/runtime/sglang"
        runtime.mkdir(parents=True)
        (runtime / "__init__.py").write_text("")
        (runtime.parent / "overlay-receipt.json").write_text("{}")
        self.config = {"host": "spark-0444", "remote_directory": "qwen27-classifier", "image": "sha256:" + "a" * 64,
                       "source_checkpoint": str(source), "model": "qwen27-classifier", "port": 30001,
                       "container": "qwen27-classifier"}
        self.write_config()

    def write_config(self):
        (self.root / "classifier.json").write_text(json.dumps(self.config))

    def test_remote_preflight_refuses_before_any_transfer(self):
        plan = deployment_plan(self.root)
        self.assertEqual(plan[0]["name"], "preflight")
        fingerprint = profile_fingerprint(self.config, b"{}")
        for profile, running in (("different", True), (fingerprint, False)):
            def execute_preflight(command, **kwargs):
                remote_argv = shlex.split(command[-1])
                inspected = [{"Config": {"Labels": {"q27.profile": profile}}, "State": {"Running": running}}]
                with patch.object(sys, "argv", ["-c", remote_argv[3]]), \
                     patch("subprocess.check_output", side_effect=["qwen27-classifier\n", json.dumps(inspected)]):
                    try:
                        exec(remote_argv[2], {})
                    except SystemExit as exc:
                        self.assertIn("preserve existing service", str(exc))
                        raise subprocess.CalledProcessError(1, command) from exc
                return Mock(returncode=0)
            with self.subTest(profile=profile, running=running), \
                 patch("deploy_classifier.subprocess.run", side_effect=execute_preflight) as run, \
                 patch("deploy_classifier.transfer_image") as transfer:
                with self.assertRaises(subprocess.CalledProcessError):
                    apply_plan(plan)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0], plan[0]["command"])
                transfer.assert_not_called()

    def test_plan_sequence_and_transfer_override(self):
        plan = deployment_plan(self.root, "192.168.2.2")
        self.assertEqual([s["name"] for s in plan], ["preflight", "create-directories", "checkpoint", "pinned-image", "runtime", "tools", "config", "start"])
        self.assertEqual(plan[2]["command"][:3], ["rsync", "-a", "--"])
        self.assertEqual(plan[2]["command"][-1], "192.168.2.2:qwen27-classifier/models/original-nvfp4/")
        self.assertIn('"$HOME/qwen27-classifier"', plan[-1]["command"][-1])
        self.assertEqual(json.loads((self.root / "classifier.json").read_text())["host"], "spark-0444")
        self.assertNotIn("--delete", str(plan))

    def test_remote_preflight_allows_same_running_profile_or_absent_container(self):
        plan = deployment_plan(self.root)
        remote_argv = shlex.split(plan[0]["command"][-1])
        fingerprint = profile_fingerprint(self.config, b"{}")
        prior = [{"Config": {"Labels": {"q27.profile": fingerprint}}, "State": {"Running": True}}]
        for names in ("", "qwen27-classifier\n"):
            def execute_preflight(command, **kwargs):
                if command == plan[0]["command"]:
                    with patch.object(sys, "argv", ["-c", remote_argv[3]]), \
                         patch("subprocess.check_output", side_effect=[names, json.dumps(prior)]) as inspect:
                        exec(remote_argv[2], {})
                        self.assertEqual(inspect.call_count, 2 if names else 1)
                return Mock(returncode=0)
            with self.subTest(names=names), patch("deploy_classifier.subprocess.run", side_effect=execute_preflight) as run:
                apply_plan(plan)
            self.assertEqual(run.call_count, 8)
            self.assertEqual(run.call_args.args[0], plan[-1]["command"])

    def test_plan_cli_never_executes(self):
        with patch.object(sys, "argv", ["deploy_classifier.py", "--root", str(self.root), "--plan"]), \
             patch("deploy_classifier.subprocess.run") as run, patch("deploy_classifier.subprocess.Popen") as popen, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(), 0)
        run.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(len(json.loads(output.getvalue())), 8)

    def test_rejects_unsafe_directory_host_and_unpinned_image(self):
        for directory in ("../oops", "/absolute", "valid/../../oops", "x;touch bad", "x space", "x/$HOME"):
            self.config["remote_directory"] = directory
            self.write_config()
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "remote_directory"):
                deployment_plan(self.root)
        self.config["remote_directory"] = "qwen27-classifier"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "target host"):
            deployment_plan(self.root, "-oProxyCommand=bad")
        self.config["image"] = "latest"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "pinned"):
            deployment_plan(self.root)

    def test_missing_prepared_runtime_has_exact_guidance(self):
        (self.root / ".scratch/runtime/overlay-receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "python3 tools/prepare_classifier.py"):
            deployment_plan(self.root)

    def test_apply_failure_stops_remaining_steps(self):
        plan = deployment_plan(self.root)
        failure = subprocess.CalledProcessError(23, plan[2]["command"])
        with patch("deploy_classifier.subprocess.run", side_effect=[Mock(returncode=0), Mock(returncode=0), failure]) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                apply_plan(plan)
        self.assertEqual(run.call_count, 3)

    def test_present_image_is_skipped(self):
        plan = deployment_plan(self.root)
        with patch("deploy_classifier.subprocess.run", return_value=Mock(returncode=0)) as run, \
             patch("deploy_classifier.transfer_image") as transfer:
            apply_plan(plan)
        self.assertEqual(run.call_count, 8)
        transfer.assert_not_called()
        self.assertEqual(run.call_args.args[0], plan[-1]["command"])

    def test_absent_image_transfers_then_continues(self):
        plan = deployment_plan(self.root)
        returns = [Mock(returncode=0)] * 3 + [Mock(returncode=1)] + [Mock(returncode=0)] * 4
        with patch("deploy_classifier.subprocess.run", side_effect=returns), patch("deploy_classifier.transfer_image") as transfer:
            apply_plan(plan)
        transfer.assert_called_once_with(plan[3]["save"], plan[3]["load"])

    def test_ssh_inspection_failure_is_not_missing_image(self):
        plan = deployment_plan(self.root)
        with patch("deploy_classifier.subprocess.run", side_effect=[Mock(returncode=0)] * 3 + [Mock(returncode=255)]) as run, \
             patch("deploy_classifier.transfer_image") as transfer:
            with self.assertRaisesRegex(RuntimeError, "inspection failed"):
                apply_plan(plan)
        self.assertEqual(run.call_count, 4)
        transfer.assert_not_called()

    def test_both_image_process_return_codes_checked(self):
        for save_code, load_code in ((0, 0), (1, 0), (0, 1), (2, 3)):
            save = Mock(stdout=Mock(), wait=Mock(return_value=save_code))
            load = Mock(wait=Mock(return_value=load_code))
            with self.subTest(save_code=save_code, load_code=load_code), \
                 patch("deploy_classifier.subprocess.Popen", side_effect=[save, load]) as popen:
                if save_code or load_code:
                    with self.assertRaisesRegex(RuntimeError, f"save exit {save_code}; remote load exit {load_code}"):
                        transfer_image(["docker", "save"], ["ssh", "load"])
                else:
                    transfer_image(["docker", "save"], ["ssh", "load"])
                self.assertIs(popen.call_args.kwargs["stdin"], save.stdout)
                save.stdout.close.assert_called_once()
                save.wait.assert_called_once()
                load.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
