import argparse
from typing import Any, Type

from megatron.training.arguments import add_megatron_arguments


def get_megatron_parser():
    """
    Extracts the arguments from megatron.training.arguments.py.
    """
    parser = argparse.ArgumentParser(description="Megatron-LM Arguments", allow_abbrev=False)
    parser = add_megatron_arguments(parser)
    return parser


def get_args_and_types(
    parser: argparse.ArgumentParser,
    exclude_args: list[str] | None = None,
    override_defaults: dict[str, Any] | None = None,
) -> dict[str, tuple[Type, Any]]:
    """
    Generates a configuration dictionary from the given arguments.

    Args:
        args: The parsed arguments from argparse.
        exclude_args: A list of argument names to exclude from the configuration.
        override_defaults: A dictionary of argument names and values to override the defaults.

    Returns:
        A dictionary representing the configuration.
    """
    args = parser.parse_args(args=[])

    config = {}
    exclude_args = exclude_args or []
    override_defaults = override_defaults or {}

    arg_types = {
        arg.dest: (bool if isinstance(arg, argparse._StoreTrueAction | argparse._StoreFalseAction) else arg.type)
        for arg in parser._actions
    }

    for arg in vars(args):
        if arg not in exclude_args:
            value = getattr(args, arg)
            if arg in override_defaults:
                value = override_defaults.get(arg, value)

            config[arg] = (arg_types[arg], value)

    return config


def _arg_to_str(a: Any):
    if a is None:
        return "null"
    else:
        return str(a)


def _arg_to_cmdline(arg: str, argval: Any, parser: argparse.ArgumentParser) -> list[str]:
    def is_default(a: argparse.Action, value: Any) -> bool:
        # Treat SUPPRESS as "no default"
        if a.default is argparse.SUPPRESS:
            return False
        return value == a.default

    cmdline = []
    for action in parser._actions:
        if action.dest == arg:
            if action.__class__.__name__ in ("_StoreConstAction",):
                if not is_default(action, argval):
                    if argval == action.const:
                        cmdline.append("--" + arg.replace("_", "-"))
            else:
                cmdline.append("--" + arg.replace("_", "-"))
                cmdline.append(_arg_to_str(argval))
            break
    return cmdline


def get_cmdline_args(
    args: dict[str, Any],
    skip_none: bool = True,
    default_skip: dict[str, Any] | None = None,
    parser: argparse.ArgumentParser | None = None,
):
    cmdline = []
    if default_skip is None:
        default_skip = {}
    for arg, val in args.items():
        if val is not None or not skip_none:
            if arg not in default_skip or default_skip[arg] != val:
                if parser is None:
                    cmdline.append("--" + arg.replace("_", "-"))
                    cmdline.append(_arg_to_str(val))
                else:
                    cmdline += _arg_to_cmdline(arg, val, parser)
    return cmdline
