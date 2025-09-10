import os
from pathlib import Path


def get_slurm_template(name: str, base_dir="../../slurm_template/"):
    if name in os.listdir(base_dir):
        with open(Path(base_dir) / name) as fp:
            return fp.read()


def generate_slurm_script(slurm_template: str, replacements: dict[str, str]):
    for key, replacement in replacements.items():
        slurm_template = slurm_template.replace("{{ " + key + " }}", replacement)
    return slurm_template
