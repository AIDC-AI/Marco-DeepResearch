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
from importlib.metadata import PackageNotFoundError, version

from packaging import version as vs


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = "vllm"
package_version = get_version(package_name)
if package_version is None:
    raise PackageNotFoundError("vLLM is required for UMEM training. Install requirements.txt first.")

vllm_version = package_version
parsed_version = vs.parse(package_version)
if parsed_version < vs.parse("0.7.0") or parsed_version > vs.parse("0.8.5"):
    raise ValueError(f"UMEM open-source training supports vLLM >=0.7.0 and <=0.8.5, got {package_version}.")

from vllm import LLM
from vllm.distributed import parallel_state

__all__ = ["LLM", "parallel_state", "vllm_version", "get_version"]
