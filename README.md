# Installation instructions

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n sal python=3.11 && conda activate sal
```

```shell
pip install -e '.[dev]'
```

Next, log into your Hugging Face account as follows:

```shell
huggingface-cli login
```

Finally, install Git LFS so that you can push models to the Hugging Face Hub:

```shell
sudo apt-get install git-lfs
```

You can now check out the `scripts` and `recipes` directories for instructions on how to scale test-time compute for open models!

## Project structure

```
├── LICENSE
├── Makefile                    <- Makefile with commands like `make style`
├── README.md                   <- The top-level README for developers using this project
├── recipes                     <- Recipe configs, accelerate configs, slurm scripts
├── scripts                     <- Scripts to scale test-time compute for models
├── pyproject.toml              <- Installation config (mostly used for configuring code quality & tests)
├── setup.py                    <- Makes project pip installable (pip install -e .) so `sal` can be imported
├── src                         <- Source code for use in this project
└── tests                       <- Unit tests
```

## Replicating our test-time compute results

The [`recipes` README](recipes/README.md) includes launch commands and config files in order to replicate our results.

# Nots on using vLLM serving
First launch the vLLM servers with launch_vllm, where you can set the number of vLLM servers, ports, and their corresponing device.
```shell
bash launch_vllm.sh
```

And then, use the following commands from the original HuggingFace repo to run the search, where you can also specify a CUDA device for the PRM, e.g.,
```shell

```
