import pandas as pd
from argparse import ArgumentParser
import os
from pathlib import Path
from typing import Literal
import numpy as np
import yaml
import doctest
import re


def apply_acc(typ: Literal["mean", "median", "max", "min"], ar: np.ndarray):
    if len(ar) == 0:
        return np.array([np.nan])
    if typ == "mean":
        return np.mean(ar)
    elif typ == "median":
        return np.median(ar)
    elif typ == "max":
        return np.max(ar)
    elif typ == "min":
        return np.min(ar)
    else:
        raise NotImplementedError


def extract_cfg(cfg: dict | int | str, key: str):
    """
    >>> extract_cfg({'a': {'b': 1, 'c': 2}}, 'a.b')
    1
    >>> extract_cfg({'a': {'b': 1, 'c': 2}}, 'a.c')
    2
    >>> extract_cfg({'a': [3, 4]}, 'a.1')
    4
    """
    if not key:
        return cfg
    key0, *keys = key.split(".")
    try:
        key0 = int(key0)
    except ValueError:
        pass
    if (isinstance(key0, int) and len(cfg) < key0) or (isinstance(key0, str) and key0 not in cfg):
        return None
    return extract_cfg(cfg[key0], ".".join(keys))


def flatten_dict(cfg: dict, sep="."):
    """
    >>> flatten_dict({"a": {"b": 1}}) == {"a.b": 1}
    True
    >>> flatten_dict({"a": [1, 2]}) == {"a.0": 1, "a.1": 2}
    True
    >>> flatten_dict({"a": 1, "b": {"c": 1}}) == {"a": 1, "b.c": 1}
    True
    """
    cfg_flat = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            for subkey, val in flatten_dict(val, sep=sep).items():
                cfg_flat[key + sep + subkey] = val
        elif isinstance(val, list | tuple):
            for idx, val in enumerate(val):
                cfg_flat.update(
                    **{key + sep + subkey: v for subkey, v in flatten_dict({str(idx): val}, sep=sep).items()}
                )
        else:
            cfg_flat[key] = val
    return cfg_flat


def main():
    parser = ArgumentParser()
    parser.add_argument("--base-dir", type=str, help="Experiments directory to get running times from")
    parser.add_argument("--exp-dir-regex", type=str, default=".*")
    parser.add_argument("--log-file", type=str, default=r".*\.out$")
    parser.add_argument("--red-type", choices=["mean", "median", "max", "min"], default="median")
    parser.add_argument("--show-failed", action="store_true")
    parser.add_argument("--cfg-file", type=str, default=r".*config\.yaml")
    parser.add_argument(
        "--extract-config",
        type=str,
        default="aux.model_name,slurmid,micro_batch_size,global_batch_size,seq_length,num_params,slurm.total_gpus",
    )

    args = parser.parse_args()

    base_dir = args.base_dir

    # col_names = []

    # res = []

    # baseline_throughput = None

    recs = []
    # cols = [
    #     "model_name",
    #     "slurm_id",
    #     "num_attention_heads",
    #     "num_nodes",
    #     "micro_batch_size",
    #     "num_gpus",
    #     "token_throughput",
    # ]
    cols = args.extract_config.split(",") + ["token_throughput"]

    print([re.match(args.exp_dir_regex, log_dir) for log_dir in os.listdir(base_dir)])

    for log_dir in os.listdir(base_dir):
        if not re.match(args.exp_dir_regex, log_dir):
            print(f"Skipped: {log_dir}")
            continue
        else:
            print(f"Taking: {log_dir}")
        exppath = Path(base_dir) / log_dir
        if os.path.isdir(exppath):
            cfgfile = [cfgfile for cfgfile in os.listdir(exppath) if re.match(args.cfg_file, cfgfile)]

            if not cfgfile:
                print(f"Missing config file in {exppath}")
                continue
            cfgfile = exppath / cfgfile[0]
            with open(cfgfile) as fp:
                cfg = yaml.safe_load(fp)

            cfg = flatten_dict(cfg, sep=".")

            res_dict = {
                key: cfg[key] if key in cfg else cfg["megatron." + key]
                for key in args.extract_config.split(",")
                if key in cfg or "megatron." + key in cfg
            }

            print(os.listdir(exppath))

            logfile = [logfile for logfile in os.listdir(exppath) if re.match(args.log_file, logfile)]
            if logfile:
                logfile = exppath / logfile[0]
                with open(logfile) as fp:
                    log = fp.read()
                try:
                    itertimes = re.findall(r"elapsed time per iteration \(ms\): (\d+\.\d+)", log)
                    itertimes = np.array([float(itertime) for itertime in itertimes])

                    num_params = re.findall(r"Total number of parameters in billions: (\d+\.\d+)", log)

                    if num_params:
                        res_dict["num_params"] = num_params[0]
                    else:
                        res_dict["num_params"] = float("nan")

                    if len(itertimes) == 0 and not args.show_failed:
                        continue

                    res_dict["itertime"] = float(apply_acc(args.red_type, itertimes))
                    res_dict["batch_size_per_device"] = res_dict["global_batch_size"] / res_dict["slurm.total_gpus"]

                    res_dict["token_throughput"] = (
                        1000 * res_dict["batch_size_per_device"] * res_dict["seq_length"] / res_dict["itertime"]
                    )
                    res_dict["slurmid"] = os.path.split(logfile)[1][:-4]
                    recs.append([res_dict[col] for col in cols])
                    print(res_dict)

                except KeyError:
                    pass

    print(recs)
    df = pd.DataFrame(data=recs, columns=cols).sort_values(
        by=["slurmid", "aux.model_name", "slurm.total_gpus"], ascending=[True, True, True]
    )

    print(df)


if __name__ == "__main__":
    doctest.testmod(verbose=False)
    main()
