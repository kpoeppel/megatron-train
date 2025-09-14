# import modelopt.torch.quantization  # noqa
from megatron_train.config import get_cmdline_args, get_args_and_types, get_megatron_parser


import argparse
import os
import yaml
from dataclasses import make_dataclass, field, dataclass, fields, MISSING
from omegaconf import OmegaConf
from pathlib import Path
from compoconf import parse_config, MissingValue, ConfigError, NonStrictDataclass, asdict
from typing import Any, Type, get_origin
from megatron_train.slurm import get_slurm_template, generate_slurm_script
from megatron_train.extract_hydra import run_hydra, oc_timestring
from megatron_train.run import run_with_tee
from megatron_train.job_log import job_log
import re

# print(get_args_and_types(get_megatron_parser()))


def _check_type(x: Any):
    try:
        # Case 1: list[...] annotation (PEP 585)
        if get_origin(x) is list:
            return True
        # Case 2: Plain type (int, str, dict, etc.)
        if isinstance(x, Type):
            return True
    except Exception:
        pass
    return False


MegatronConfig = make_dataclass(
    "MegatronConfig",
    [
        (
            arg,
            (argtype if _check_type(argtype) else (str | int)) | None if argtype is not None else (str | None),
            field(default=defval) if not isinstance(defval, list) else field(default_factory=lambda: defval),
        )
        for arg, (argtype, defval) in get_args_and_types(get_megatron_parser()).items()
    ]
    + [("aux", dict[str, Any], field(default_factory=dict))],
)


def mcfg_post_init(self):
    assert self.micro_batch_size is not None
    # assert self.global_batch_size is not None  # may be set at higher level by gpus
    assert self.train_iters is not None
    assert self.lr_decay_style is not None
    assert self.lr is not None
    assert self.num_layers is not None
    assert self.hidden_size is not None
    assert self.ffn_hidden_size is not None
    assert self.kv_channels is not None
    assert self.num_attention_heads is not None
    assert self.vocab_size is not None
    assert self.max_position_embeddings is not None

    assert not (self.use_distributed_optimizer and self.use_torch_fsdp2)
    assert not (self.use_megatron_fsdp and self.use_torch_fsdp2)

    assert self.lr_warmup_iters <= self.train_iters

    assert not (self.use_megatron_fsdp and self.ckpt_format != "fsdp_dtensor")
    assert not (self.overlap_param_gather and not (self.use_distributed_optimizer or self.use_megatron_fsdp))

    assert not (self.gradient_accumulation_fusion and self.use_torch_fsdp2)

    assert not (self.gradient_accumulation_fusion and self.data_parallel_sharding_strategy not in ["no_shard", "optim"])


MegatronConfig.__post_init__ = mcfg_post_init


def quote_bash(st: str):
    return st.replace("'", "'\"'\"'")


@dataclass(init=False)
class SlurmConfig(NonStrictDataclass):
    nodes: int = 1
    gpus_per_node: int = 1
    account: str = MissingValue
    partition: str = MissingValue
    total_gpus: int = MissingValue
    template: str = MissingValue

    def __post_init__(self):
        if self.template is MissingValue:
            raise ConfigError(str(self))
        if self.total_gpus is MissingValue:
            self.total_gpus = self.num_nodes * self.gpus_per_node
        assert self.total_gpus == self.nodes * self.gpus_per_node

        assert not any(getattr(self, f.name) is MissingValue or getattr(self, f.name) == "" for f in fields(self))


@dataclass(init=False)
class LauncherConfig(NonStrictDataclass):
    cmd: str = MISSING


@dataclass(init=False)
class SRunConfig(NonStrictDataclass):
    opts: str = MISSING


@dataclass(init=False)
class MegatronTrainConfig(NonStrictDataclass):
    megatron: MegatronConfig = field(default_factory=MegatronConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    env: dict[str, str | int | float | None] = field(default_factory=dict)
    launcher: LauncherConfig = field(default=LauncherConfig)
    srun: SRunConfig = field(default=SRunConfig)

    global_batch_size: int = 1
    experiment_name: str = "debug"

    timestamp: str = field(default_factory=oc_timestring)
    output_dir: str = field(default_factory=MISSING)
    nest_launcher: bool = True

    def __post_init__(self):
        # CUDA_DEVICE_MAX_CONNECTIONS must be >= 1 with fsdp
        assert not (
            "CUDA_DEVICE_MAX_CONNECTIONS" in self.env
            and int(self.env["CUDA_DEVICE_MAX_CONNECTIONS"]) <= 1
            and self.megatron.data_parallel_sharding_strategy != "no_shard"
        )


def slurm_script_from_config(config: MegatronTrainConfig, cmdline_args: list[str]) -> str:
    print(config.slurm)
    slurm_template = get_slurm_template(config.slurm.template, base_dir="./slurm_template")

    sbatch_cmds = "\n".join(
        [
            f"#SBATCH --{k.replace('_', '-')}={v}"
            for k, v in asdict(config.slurm).items()
            if k not in ["template", "total_gpus", "_non_strict"]
        ]
    )

    env_exports = "\n".join(["export " + k + "=" + str(v) for k, v in config.env.items()])

    launcher = config.launcher.cmd

    srun_opts = config.srun.opts

    megatron_cmd = " ".join(["$RUN_DIR/Megatron-LM/pretrain_gpt.py"] + cmdline_args)

    if config.nest_launcher:
        launcher = quote_bash(launcher)
        megatron_cmd = quote_bash(megatron_cmd)

    slurm_script = generate_slurm_script(
        slurm_template,
        {
            "env_exports": env_exports,
            "sbatch_cmds": sbatch_cmds,
            "launcher": launcher,
            "srun_opts": srun_opts,
            "megatron_cmd": megatron_cmd,
        },
    )

    return slurm_script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="./config", help="Path to config directory")
    parser.add_argument("--config-name", type=str, default="base", help="Name of base config file")
    parser.add_argument("--config-name-default", type=str, default="base", help="Name of base config file")
    parser.add_argument("--config-yaml", type=str, default="", help="Additional YAML config to override")
    # parser.add_argument("--no-diff-to-default", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--show-log", action="store_true")

    parser.add_argument(
        "opts",
        nargs="*",
        default=[],
        help="Additional arguments to override config (e.g. dataset.local_batch_size=32)",
    )
    args = parser.parse_args()

    config_yaml = run_hydra(
        config_path=args.config_path,
        config_name=args.config_name,
        cmdline_opts=args.opts,
        config_yaml=args.config_yaml,
    )
    config_yaml_base = yaml.safe_load(config_yaml)
    config = OmegaConf.create(config_yaml_base)

    OmegaConf.resolve(config)
    config = OmegaConf.to_container(config)
    config = parse_config(MegatronTrainConfig, config)

    # config_yaml_default = run_hydra(
    #     config_path=args.config_path,
    #     config_name=args.config_name_default,
    # )
    # config_yaml_base = yaml.safe_load(config_yaml_default)
    # config_default = OmegaConf.create(config_yaml_base)
    # OmegaConf.resolve(config_default)
    # config_default = OmegaConf.to_container(config_default)
    # config_default = parse_config(MegatronTrainConfig, config_default)

    cmdline_args = get_cmdline_args(
        asdict(config.megatron),
        skip_none=True,
        ignore_args=["aux"],
        default_skip={},  # asdict(config_default.megatron) if not args.no_diff_to_default else {},
        parser=get_megatron_parser(),
    )
    # print("MEGATRON CMDLINE:", cmdline_args)

    slurm_script = slurm_script_from_config(config, cmdline_args)

    if args.debug:
        print(f"Output Directory: {config.output_dir}")
        print("SLURM_SCRIPT:")
        print(slurm_script)
    else:
        os.makedirs(config.output_dir)
        print(f"Output Directory: {config.output_dir}")
        print(f"SLURMOUT: {config.slurm.output}")

        with open(Path(config.output_dir) / "train_megatron.sbatch", "w") as fp:
            fp.write(slurm_script)
        with open(Path(config.output_dir) / "submit_config.yaml", "w") as fp:
            yaml.dump(asdict(config), fp)

        if args.run:
            out = run_with_tee(["sbatch", str(Path(config.output_dir) / "train_megatron.sbatch")], text=True)
            if args.show_log:
                match = re.search(r"Submitted batch job (\d+)", out.stdout, flags=re.MULTILINE)
                if match:
                    jobid = match.group(1)
                    job_log(jobid)
        else:
            print(
                f"Successful, to execute, run: SUBMIT_TIMESTAMP={config.timestamp} sbatch {str(Path(config.output_dir) / 'train_megatron.sbatch')}"
            )


if __name__ == "__main__":
    main()
