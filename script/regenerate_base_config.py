import yaml
from megatron_train.config import get_megatron_parser, get_args_and_types, get_choices_arg, get_help
import argparse
import re
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
    parser.add_argument("--base-config-file", default="config/megatron/base_empty.yaml")
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
    megatron_config = {
        k: str(v[1]).split(".")[-1] if isinstance(v[1], enum.Enum) else v[1] for k, v in megatron_config.items()
    }

    megatron_cfgyaml = yaml.dump(megatron_config)

    # add comments on options
    for key in megatron_config:
        choices = get_choices_arg(megatron_parser, key)
        helpstr = get_help(megatron_parser, key)
        if choices is not None or helpstr:
            match = re.search(key + ":" + ".*", megatron_cfgyaml, flags=re.MULTILINE)
            if match:
                megatron_cfgyaml = (
                    megatron_cfgyaml[: match.end()]
                    + "  # "
                    + (f"choices: {choices}, " if choices else "")
                    + f"{helpstr}"
                    + megatron_cfgyaml[match.end() :]
                )

    with open(args.base_config_file, "w") as fp:
        fp.write(megatron_cfgyaml)


if __name__ == "__main__":
    main()
