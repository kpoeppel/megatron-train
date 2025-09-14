#!/bin/python

import subprocess
import argparse
import os
from pathlib import Path
import re
from megatron_train.run import run_with_tee
from megatron_train.job_log import job_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
    )
    parser.add_argument(
        "--apptainer-cmd",
        type=str,
        default="singularity",
    )
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument(
        "--env",
        default="PYTHONPATH=src:Megatron-LM",
    )
    parser.add_argument("--show-log", action="store_true")
    parser.add_argument("--ihelp", action="store_true", help="Help on run_megatron.py")
    args, other = parser.parse_known_args()

    if args.ihelp:
        other.append("--help")

    run_megatron_file = Path(os.path.split(os.path.abspath(__file__))[0]) / "run_megatron.py"

    res = subprocess.run(
        [
            args.apptainer_cmd,
            "exec",
            args.image,
            "bash",
            "-c",
            f"{args.env} python {run_megatron_file} " + " ".join(other),
        ],
        capture_output=not args.ihelp,
    )

    if not args.ihelp:
        print(f"STDOUT {res.returncode}", res.stdout.decode("utf-8"))
        print("ERRORS: ", res.stderr.decode("utf-8"))
        if res.returncode == 0:
            sbatch_cmd = re.search(
                "^(Successful, to execute, run: )(.*)(sbatch .*)", res.stdout.decode("utf-8"), flags=re.MULTILINE
            )
            print(res.stdout.decode("utf-8"))
            if sbatch_cmd:
                slurm_env = sbatch_cmd.group(2)
                slurm_cmd = sbatch_cmd.group(3)
                print(f"Slurm Command: {slurm_cmd}")
                if not args.no_run:
                    env = dict(**os.environ)
                    env_subst = {}
                    for env_change in slurm_env.split(" "):
                        if "=" in env_change:
                            env_subst[env_change.split("=")[0]] = env_change.split("=")[1]
                    env.update(**env_subst)
                    print(f"Submit: {slurm_cmd} with env {env_subst}")
                    out = run_with_tee(slurm_cmd.split(" "), env=env, text=True)
                    if args.show_log:
                        match = re.search(r"Submitted batch job (\d+)", out.stdout, flags=re.MULTILINE)
                        if match:
                            jobid = match.group(1)
                            job_log(jobid)
            else:
                print("Error finding submit command")


if __name__ == "__main__":
    main()
