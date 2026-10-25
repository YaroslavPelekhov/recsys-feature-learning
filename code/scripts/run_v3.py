"""One-process entry point for the complete v3 study and its secondary extension."""

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def command(script, *arguments):
    return [sys.executable, str(Path("scripts") / script), *map(str, arguments)]


def run(script, *arguments):
    subprocess.run(command(script, *arguments), check=True)


def record_runtime(root):
    if (root / "neural_environment.json").exists():
        return
    import psutil
    import torch

    from rec.io import write_json

    cuda = torch.cuda.is_available()
    write_json(
        root / "neural_environment.json",
        {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name() if cuda else "CPU",
            "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else 0,
            "cpu": platform.processor(),
            "ram_bytes": psutil.virtual_memory().total,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/v3-reproduction"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/v3-reproduction"))
    parser.add_argument("--without-support", action="store_true")
    args = parser.parse_args()
    root = args.root
    support = root.with_name(root.name + "_support")
    support_report = args.report_dir.with_name(args.report_dir.name + "_support")
    run("download_yambda.py", "--root", root)
    record_runtime(root)
    if not (root / "frozen_recipe.json").exists():
        run("parallel_v3.py", "development", "--root", root)
    elif not (root / "confirm/results.json").exists():
        run("parallel_v3.py", "confirm", "--root", root)
    if not args.without_support:
        if not (support / "frozen_recipe.json").exists():
            run("support_v3.py", "development", "--base", root, "--root", support)
        if not (support / "noise_diagnostics.json").exists():
            run(
                "diagnose_noise_support.py",
                "--base",
                root,
                "--output",
                support / "noise_diagnostics.json",
            )
        if not (support / "neighbor_audit.json").exists():
            run(
                "audit_neighbors_v3.py",
                "--base",
                root,
                "--support",
                support,
                "--output",
                support,
            )
        if not (support / "critic_audit.json").exists():
            run("audit_critic_v3.py", "--base", root, "--support", support, "--output", support)
    if not (root / "test/complete.json").exists():
        pending = None
        try:
            if not args.without_support and not (support / "predictions_ready.json").exists():
                pending = subprocess.Popen(
                    command("support_v3.py", "predict", "--base", root, "--root", support),
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
            arguments = ["test", "--root", root]
            if not args.without_support:
                arguments.extend(["--support-root", support])
            run("parallel_v3.py", *arguments)
            if pending is not None and pending.wait() != 0:
                raise RuntimeError("Secondary prediction process failed")
        finally:
            if pending is not None and pending.poll() is None:
                pending.terminate()
                pending.wait()
    run("audit_exposure_v3.py", "--run", root)
    if not args.without_support:
        if not (support / "test/complete.json").exists():
            run("support_v3.py", "evaluate", "--base", root, "--root", support)
        run("report_v3_support.py", "--base", root, "--run", support, "--output", support_report)
    run("replay_v3.py", "--run", root)
    run("report_v3.py", "--run", root, "--output", args.report_dir)


if __name__ == "__main__":
    main()
