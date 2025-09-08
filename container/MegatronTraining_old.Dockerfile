FROM nvcr.io/nvidia/pytorch:25.08-py3

RUN apt-get update --yes --quiet \
    && apt-get upgrade --yes --quiet \
    && apt-get install tmux vim git nano wget \
    curl unzip software-properties-common tmux \
    strace iproute2 --yes --quiet

WORKDIR /workspace

COPY gen_constraints_from_pip_freeze.py /workspace/gen_constraints_from_pip_freeze.py

COPY ../Megatron-LM /workspace/Megatron-LM

RUN pip freeze

RUN pip install --upgrade pip setuptools 
RUN pip freeze | python gen_constraints_from_pip_freeze.py > constraints.txt
RUN cat constraints.txt

RUN pip install -c constraints.txt ninja packaging psutil wheel appdirs>=1.4.4 fire>=0.6.0 multipledispatch>=0.6.0 pqdm>=0.2.0 \
    tempdir>=0.7.1 termcolor>=2.0.0 tqdm>=4.56.0 tree_sitter_languages>=1.10.2 tree-sitter==0.21.3 wget pudb

RUN pip install --no-cache-dir -c constraints.txt fvcore torchtnt torchtune torchao matplotlib tensorboard \
    tqdm transformers pandas 'torchdata>=0.8.0' 'datasets>=3.6.0' 'tomli>=1.1.0' \
    tiktoken blobfile tabulate wandb fsspec tyro expecttest pudb \
    'pytest<=8.3.0' pytest-cov pytest-pudb pre-commit lm_eval \
    dacite compoconf datasets scipy safetensors tabulate tensorboard tensorflow einops \
    compoconf hydra-core opt-einsum pandas

RUN pip install --no-cache-dir -c constraints.txt -e /workspace/Megatron-LM
RUN pip uninstall megatron-core