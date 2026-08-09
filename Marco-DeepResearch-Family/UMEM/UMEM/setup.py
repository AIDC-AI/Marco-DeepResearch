# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
with open(os.path.join(ROOT, "verl/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate",
    "aiohttp",
    "codetiming",
    "datasets",
    "dill",
    "faiss-cpu",
    "hydra-core",
    "numpy",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0",
    "tensordict<=0.6.2",
    "torchdata",
    "transformers",
    "wandb",
    "packaging>=20.0",
]

extras_require = {
    "sglang": [
        "sglang[all]==0.4.6.post5",
        "torch-memory-saver>=0.0.5",
        "huggingface_hub",
    ],
    "vllm": ["vllm>=0.7.0,<=0.8.5", "flash-attn", "liger-kernel"],
}

setup(
    name="umem",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    license="Apache 2.0",
    author="UMEM contributors",
    description="UMEM: memory-augmented reinforcement learning for LLMs",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml"],
    },
    include_package_data=True,
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
)
