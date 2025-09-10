from megatron_train.extract_hydra import run_hydra
from megatron_train.config import get_cmdline_args, get_args_and_types, get_megatron_parser
import argparse
from omegaconf import OmegaConf
import yaml
from dataclasses import make_dataclass, field, dataclass, is_dataclass, MISSING
from compoconf import parse_config, MissingValue, ConfigError, NonStrictDataclass, asdict
from typing import Callable
from megatron_train.slurm import get_slurm_template, generate_slurm_script

print(get_args_and_types(get_megatron_parser()))


MegatronConfig = make_dataclass(
    "MegatronConfig",
    [
        (
            arg,
            (argtype if not isinstance(argtype, Callable) else (str | int)) | None
            if argtype is not None
            else (str | None),
            field(default=defval) if not isinstance(defval, list) else field(default_factory=lambda: defval),
        )
        for arg, (argtype, defval) in get_args_and_types(get_megatron_parser()).items()
    ],
)


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
            raise ConfigError
        if self.total_gpus is MissingValue:
            self.total_gpus = self.num_nodes * self.num_gpus_per_node
        raise self.total_gpus == self.num_nodes * self.num_gpus_per_node

        assert any(f.value is MissingValue or f.value == "" for f in self.fields)


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


def slurm_script_from_config(config: MegatronTrainConfig, cmdline_args: list[str]) -> str:
    print(config.slurm)
    slurm_template = get_slurm_template(config.slurm.template, base_dir="./slurm_template")

    sbatch_cmds = "\n".join(
        [
            f"#SBATCH: --{k}={v}"
            for k, v in asdict(config.slurm).items()
            if k not in ["template", "total_gpus", "_non_strict"]
        ]
    )

    env_exports = "\n".join(["export " + k + "=" + str(v) for k, v in config.env.items()])

    launcher = config.launcher.cmd

    srun_opts = config.srun.opts

    megatron_cmd = " ".join(["$RUN_DIR/Megatron-LM/pretrain_gpt.py"] + cmdline_args)

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

    print(slurm_script)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="./config", help="Path to config directory")
    parser.add_argument("--config-name", type=str, default="base", help="Name of base config file")
    parser.add_argument("--config-name-default", type=str, default="base", help="Name of base config file")
    parser.add_argument("--config-yaml", type=str, default="", help="Additional YAML config to override")
    parser.add_argument("--no-diff-to-default", action="store_true")

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

    print(config)

    OmegaConf.resolve(config)
    config = OmegaConf.to_container(config)
    config = parse_config(MegatronTrainConfig, config)

    config_yaml_default = run_hydra(
        config_path=args.config_path,
        config_name=args.config_name_default,
    )
    config_yaml_base = yaml.safe_load(config_yaml_default)
    config_default = OmegaConf.create(config_yaml_base)
    OmegaConf.resolve(config_default)
    config_default = OmegaConf.to_container(config_default)
    config_default = parse_config(MegatronTrainConfig, config_default)

    cmdline_args = get_cmdline_args(
        asdict(config.megatron),
        skip_none=True,
        default_skip=asdict(config_default.megatron) if not args.no_diff_to_default else {},
    )

    print("MEGATRON CMDLINE:", cmdline_args)

    slurm_script_from_config(config, cmdline_args)


if __name__ == "__main__":
    main()
