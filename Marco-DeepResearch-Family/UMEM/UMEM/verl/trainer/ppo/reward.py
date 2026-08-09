# Copyright 2025 Individual Contributor: Thibaut Barroyer
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

import ray

from verl import DataProto


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """Load the UMEM reward manager configured for train or validation."""
    from verl.workers.reward_manager import get_reward_manager_cls

    key = "reward_manager" if num_examine == 0 else "val_reward_manager"
    reward_manager_name = config.reward_model.get(key, "extractor")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=None,
        reward_fn_key=config.data.reward_fn_key,
        config=config,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn):
    """Compute reward tensor and optional reward metadata."""
    reward_result = reward_fn(data, return_dict=True)
    reward_tensor = reward_result["reward_tensor"]
    reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)
