import yaml
from train_megatron.config import get_megatron_parser, get_args_and_types
import argparse
import enum


def to_value(obj_str: str):
    try:
        return int(obj_str)
    except ValueError:
        try:
            return float(obj_str)
        except ValueError:
            return obj_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config-file", default="config/megatron/base.yaml")
    parser.add_argument("--exclude-args", type=str, default="")
    parser.add_argument("--override-defaults", nargs="*", default=[])

    args = parser.parse_args()

    override_defaults = {
        override.split("=")[0]: to_value(override.split("=")[1]) for override in args.override_defaults
    }

    megatron_parser = get_megatron_parser()
    megatron_config = get_args_and_types(
        megatron_parser, exclude_args=args.exclude_args.split(","), override_defaults=override_defaults
    )
    megatron_config = {k: v[1].value if isinstance(v[1], enum.Enum) else v[1] for k, v in megatron_config.items()}

    with open(args.base_config_file, "w") as fp:
        yaml.dump(megatron_config, fp)


if __name__ == "__main__":
    main()
