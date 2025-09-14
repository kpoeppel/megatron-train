import re
import time
import signal
import atexit
import os
from .run import run_with_tee


def job_log(jobid: str):
    out = run_with_tee(["scontrol", "show", f"jobid={jobid}"], text=True)
    match = re.search(r"StdOut=([^\s]+)", out.stdout, flags=re.MULTILINE)
    if match:
        outfile = match.group(1)
        n = 0
        while not os.path.exists(outfile):
            print("\b" * 100 + f"Waiting for job {jobid} to be started.", end="")
            n += 1
            time.sleep(1)
        print("\n")
        # doesn't work as history is a builti-n
        # run_with_tee(["history", "-s", "tail", "-n", "100000", "-f", outfile], text=True)
        log_cmd = ["tail", "-n", "100000", "-f", outfile]

        @atexit.register
        def print_log_cmd():
            print("\n\n" + " ".join(log_cmd))

        def signal_handler(signum, frame):
            print_log_cmd()
            os._exit(128 + signum)

        for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(_sig, signal_handler)
            except Exception:
                pass

        run_with_tee(log_cmd, text=True)
