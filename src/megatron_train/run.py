import subprocess
import sys
import threading
from typing import Sequence, Union, Optional


def run_with_tee(
    args: Union[str, Sequence[str]],
    *,
    check: bool = False,
    timeout: Optional[float] = None,
    input: Optional[Union[str, bytes]] = None,
    text: Optional[bool] = None,  # None = follow Python default; True = text mode; False = bytes
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    env=None,
    cwd=None,
    shell: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a command like subprocess.run but:
      - streams stdout/stderr live to this process' stdout/stderr (tee)
      - returns a CompletedProcess with captured stdout/stderr
      - supports check, timeout, input, text/encoding/errors

    NOTE: We always pipe child stdout/stderr to implement tee behavior.
    """

    popen_kwargs = {
        "stdin": subprocess.PIPE if input is not None else None,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": env,
        "cwd": cwd,
        "shell": shell,
        "text": text,
        "encoding": encoding,
        "errors": errors,
    }

    proc = subprocess.Popen(args, **popen_kwargs)

    # Select console targets and buffers depending on text/binary mode
    is_text = proc.stdout is not None and (text or encoding or errors) is not None or (text is True)
    if is_text is None:  # follow Popen's actual mode
        is_text = proc.text if hasattr(proc, "text") else False

    out_buf = []
    err_buf = []

    def _forward(src, sink, buf, chunk_bytes: int = 8192):
        if src is None:
            return
        if is_text:
            for line in iter(src.readline, ""):
                buf.append(line)
                sink.write(line)
                sink.flush()
        else:
            # binary mode: forward in chunks
            bsink = sink.buffer if hasattr(sink, "buffer") else sink
            for chunk in iter(lambda: src.read(chunk_bytes), b""):
                buf.append(chunk)
                bsink.write(chunk)
                bsink.flush()
        src.close()

    t_out = threading.Thread(target=_forward, args=(proc.stdout, sys.stdout, out_buf))
    t_err = threading.Thread(target=_forward, args=(proc.stderr, sys.stderr, err_buf))
    t_out.start()
    t_err.start()

    # Write input (if any)
    if input is not None:
        try:
            if is_text or isinstance(input, str):
                data = input if isinstance(input, str) else input.decode(encoding or "utf-8", errors or "strict")
            else:
                data = input if isinstance(input, (bytes, bytearray)) else str(input).encode()
            proc.stdin.write(data)  # type: ignore[union-attr]
        finally:
            proc.stdin.close()  # type: ignore[union-attr]

    try:
        retcode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # Kill, finish draining, then raise with partial output
        proc.kill()
        proc.wait()
        t_out.join()
        t_err.join()
        captured_stdout = "".join(out_buf) if is_text else b"".join(out_buf)
        captured_stderr = "".join(err_buf) if is_text else b"".join(err_buf)
        e.output = captured_stdout
        e.stderr = captured_stderr
        raise

    t_out.join()
    t_err.join()

    captured_stdout = "".join(out_buf) if is_text else b"".join(out_buf)
    captured_stderr = "".join(err_buf) if is_text else b"".join(err_buf)

    if check and retcode != 0:
        raise subprocess.CalledProcessError(retcode, args, output=captured_stdout, stderr=captured_stderr)

    return subprocess.CompletedProcess(
        args=args,
        returncode=retcode,
        stdout=captured_stdout,
        stderr=captured_stderr,
    )
