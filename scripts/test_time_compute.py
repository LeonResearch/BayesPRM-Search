#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import argparse
from vllm import LLM
from openai import OpenAI

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}

def send_message(client, instruction, config):
    response = client.chat.completions.create(
        model=config.model_path,
        messages=instruction,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
    )
    return response.choices[0].message.content


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    # Setup the vLLM servers
    api_key = "EMPTY"
    #api_bases = [f"http://localhost:{config.port+i}/v1" for i in range(config.n_clients)]
    api_bases = [f"http://0.0.0.0:{config.port+i}/v1" for i in range(config.n_clients)]

    clients = [OpenAI(base_url=api_bases[i], api_key=api_key) for i in range(config.n_clients)]
    # Setup the dataset and assign each datapoint a client for parallel purposes
    dataset = get_dataset(config)
    client_ids = [i % config.n_clients for i in range(len(dataset))]
    dataset = dataset.add_column("client_id", client_ids)

    approach_fn = APPROACHES[config.approach]
    num_gpus = torch.cuda.device_count()

    print("%%%%%% Loading the inference LLM ... %%%%%%")
    '''
    llm = LLM(
        model=config.model_path,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=config.seed,
        # tensor_parallel_size=num_gpus,
    )
    '''
    print("%%%%%% Inference LLM Loaded! %%%%%%")
    print("%%%%%% Loading the reward model ... %%%%%%")
    prm = load_prm(config)
    print("%%%%%% PRM Loaded! %%%%%%")

    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config":config, "prm":prm, "clients":clients},
        desc="Running search",
        load_from_cache_file=False,
    )

    dataset = score(dataset, config)

    save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
