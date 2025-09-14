#!/bin/bash

if [[ -z "$1" ]]
then
echo "Usage: ./build_container.sh DEFFILE REQUIREMENTS.txt [--append_date]  (DEFFILE without .def ending)"
else
module load GCC 
if [[ $(module keywords Singularity-Tools 2>&1 >/dev/null | grep -o Singularity-Tools | wc -l) -gt 1 ]]
then
    module load Singularity-Tools
fi
if [[ $(module keywords Apptainer-Tools 2>&1 >/dev/null | grep -o Apptainer-Tools | wc -l) -gt 1 ]]
then
    module load Apptainer-Tools
fi

export ARCH=$(uname -m)

if [[ -z "$CONTAINER_CACHE_DIR" ]]
then
CONTAINER_CACHE_DIR=$(pwd)
fi

envsubst < "$1.def.in" > "$1_$ARCH.def"

sed -i 's;{{ FROZEN_REQUIREMENTS_FILE }} /workspace/{{ FROZEN_REQUIREMENTS_FILE }};;' "$1_$ARCH.def"

if [[ -f $(grep -m 1 'From: ' $1_$ARCH.def | sed 's,From: ,,') ]]
then
echo "Base cached image already exists."
else
python extract_base_image.py "$1.def.in"
fi

if [[ $3 == "--append-date" ]] || [[ $2 == "--append-date" ]]
then
APPDATE="_$(date +%Y%m%d%H%M)"
else
APPDATE=""
fi

if [[ $2 == "--append-date" ]]
then
FROZEN_REQUIREMENTS_FILE=""
else
FROZEN_REQUIREMENTS_FILE=$2
fi

apptainer build --build-arg FROZEN_REQUIREMENTS_FILE=$FROZEN_REQUIREMENTS_FILE "$CONTAINER_CACHE_DIR/$1_$ARCH$APPDATE.sif" "$1_$ARCH.def"

apptainer exec "$CONTAINER_CACHE_DIR/$1_$ARCH$APPDATE.sif" /bin/bash -c 'pip freeze | grep -v @ | grep -vE '"'"^-e"'"' | grep -vE '"'"'^\s*#'"'"' > '"$1_$ARCH$APPDATE"'.txt'
fi