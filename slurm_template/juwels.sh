#!/bin/bash
{{ sbatch_cmds }}

{{ env_exports }}

# export MASTER_ADDR_NAME="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i"
# export MASTER_ADDR=$(nslookup $MASTER_ADDR_NAME | grep "Address: " | tail -n1 | awk '{print $2}' )
# export MASTER_PORT=20073

# echo "MASTER_ADDR" $MASTER_ADDR_NAME $MASTER_ADDR $MASTER_PORT
# nslookup $MASTER_ADDR_NAME



ml Stages/2025
ml GCC/13.3.0
ml Python/3.12.3
ml CUDA/12
ml cuDNN/9.5.0.50-CUDA-12
ml NCCL/default-CUDA-12

ml OpenMPI/5.0.5
ml ParaStationMPI/5.11.0-1
ml ParaStationMPI/5.11.0-1-mt
ml ScaLAPACK/2.2.0-fb

# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )

# for (( i=0; i<$SLURM_NNODES; i++ )); do
#     node=${nodes[$i]}
# -w ${node}
srun {{ srun_opts }} bash -c 'echo $SLURM_PROCID ; {{ launcher }} {{ megatron_cmd }}' # &

#     if [ $i -eq 0 ]; then
#         sleep 10  # Master needs time to start
#     else
#         sleep 2
#     fi
# done

# wait