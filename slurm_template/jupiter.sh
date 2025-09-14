#!/bin/bash

{{ sbatch_cmds }}

{{ env_exports }}


srun {{ srun_opts }} bash -c '{{ launcher }} {{ megatron_cmd }}'