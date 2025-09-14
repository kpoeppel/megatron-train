#!/usr/bin/env python3
"""
Script to extract base image from Singularity definition files and create a minimal .def file
for caching the base image as a local .sif file.

Usage:
    python extract_base_image.py <input_def_file> [output_def_file]

If output_def_file is not specified, it will be named as <input_basename>_base.def
"""

import sys
import os
import re
import argparse
from pathlib import Path
import subprocess


def substitute_env_variables(content):
    """Substitute environment variables in the content using ${VAR} syntax."""

    def replace_var(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))  # Return original if env var not found

    # Replace ${VAR} patterns
    content = re.sub(r"\$\{([^}]+)\}", replace_var, content)
    return content


def extract_base_image_info(def_file_path):
    """
    Extract the base image information from a Singularity definition file.

    Args:
        def_file_path (str): Path to the .def or .def.in file

    Returns:
        tuple: (bootstrap_type, base_image, created_image) or (None, None) if not found
    """
    try:
        with open(def_file_path, "r") as f:
            content = f.read()

        # If it's a .def.in file, substitute environment variables
        if def_file_path.endswith(".def.in"):
            content = substitute_env_variables(content)

        lines = content.split("\n")

        bootstrap_type = None
        base_image = None

        for i, line in enumerate(lines):
            line = line.strip()

            if line.startswith("Bootstrap: localimage"):
                next_line = lines[i + 1].strip()
                if next_line.startswith("From: "):
                    created_image = next_line[6:]
                else:
                    created_image = "image.sif"

            # Look for commented Bootstrap: docker line
            if line.startswith("# Bootstrap:") and "docker" in line.lower():
                bootstrap_type = "docker"

                # Look for the next line which should contain the From: directive
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith("# From:"):
                        base_image = next_line.replace("# From:", "").strip()
                        break

        return bootstrap_type, base_image, created_image

    except FileNotFoundError:
        print(f"Error: File '{def_file_path}' not found.")
        return None, None, None
    except Exception as e:
        print(f"Error reading file '{def_file_path}': {e}")
        return None, None, None


def create_base_def_file(bootstrap_type, base_image, output_path):
    """
    Create a minimal Singularity definition file for the base image.

    Args:
        bootstrap_type (str): The bootstrap type (e.g., 'docker')
        base_image (str): The base image identifier
        output_path (str): Path where to save the output .def file
    """

    # Extract image name for labeling
    image_name = base_image.split("/")[-1] if "/" in base_image else base_image

    def_content = f"""Bootstrap: {bootstrap_type}
From: {base_image}

%labels
    Author OpenSci
    Version 1.0.0
    Base {base_image}
    Description Minimal base image cache for {image_name}

%runscript
    exec /bin/bash "$@"

%startscript
    exec /bin/bash "$@"
"""

    try:
        with open(output_path, "w") as f:
            f.write(def_content)
        print(f"Created base definition file: {output_path}")
        print(f"Bootstrap: {bootstrap_type}")
        print(f"From: {base_image}")
        return True
    except Exception as e:
        print(f"Error writing to '{output_path}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract base image from Singularity definition files and create minimal .def file for caching"
    )
    parser.add_argument("input_file", help="Input .def or .def.in file")
    parser.add_argument("output_file", nargs="?", help="Output .def file (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    input_path = args.input_file

    # Generate output filename if not provided
    if args.output_file:
        output_path = args.output_file
    else:
        input_stem = Path(input_path).stem
        # Remove .def.in or .def extension and add _base.def
        if input_stem.endswith(".def"):
            input_stem = input_stem[:-4]
        output_path = str(Path(input_path).parent / f"{input_stem}_base.def")

    if args.verbose:
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")

    # Extract base image information
    bootstrap_type, base_image, created_image = extract_base_image_info(input_path)

    if bootstrap_type is None or base_image is None:
        print("Error: Could not find commented '# Bootstrap: docker' and '# From:' lines in the definition file.")
        print("Make sure the file contains lines like:")
        print("# Bootstrap: docker")
        print("# From: nvcr.io/nvidia/pytorch:25.08-py3")
        sys.exit(1)

    # Create the base definition file
    success = create_base_def_file(bootstrap_type, base_image, output_path)

    if not success:
        sys.exit(1)

    cmd = [
        "singularity",
        "build",
        created_image,
        output_path,
    ]

    print(" ".join(cmd))
    res = subprocess.run(cmd)

    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
