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
import logging
import re
from typing import Any

import datasets

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters["code"]
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        # sandbox has no score or metrics, use Nones
        return result, None, None


answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset(parquet_file)["train"]
            data_source = "/".join(parquet_file.split("/")[-2:])
            if data_source in ["Maxwell-Jia/AIME_2024", "yentinglin/aime_2025"]:
                dataframe = dataframe.map(
                    self.map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
                )
            else:
                dataframe = dataframe.map(self.map_fn2, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict, *, data_source: str = None):
        if data_source == "Maxwell-Jia/AIME_2024":
            problem, answer = row["Problem"], row["Answer"]
        elif data_source == "yentinglin/aime_2025":
            problem, answer = row["problem"], row["answer"]

        prompt = problem + answer_format
        data = {
            "data_source": data_source.split("/")[1].lower(),  # aime_2024, aime_2025
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",
        }
        return data

    def map_fn2(self, row: dict):
        content = row["prompt"][0]["content"]
        row["prompt"][0]["content"] = content + answer_format
        row["agent_name"] = "tool_agent"
        return row


import datasets

class CustomRLHFDataset_to_test(RLHFDataset):
    """
    Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets.
    Compatible with legacy datasets (like DAPO) via fallback branch.
    """

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            try:
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            except FileNotFoundError:
                print(f"❌ 严重错误: 找不到文件 {parquet_file}")
                print("提示: 如果这是一个文件夹，请在路径后加上 '/*.parquet'")
                raise
            cols = dataframe.column_names

            # 如果包含 AIME 特有的 problem 和 answer 列 -> 走 AIME 处理逻辑
            if "problem" in cols and "answer" in cols:
                print(f"检测到 AIME 2025 格式: {parquet_file}")
                dataframe = dataframe.map(
                    self.map_fn,
                    fn_kwargs={"data_source": "yentinglin/aime_2025"},
                    remove_columns=cols
                )

            # 兼容旧版 AIME 2024 (首字母大写)
            elif "Problem" in cols and "Answer" in cols:
                print(f"检测到 AIME 2024 格式: {parquet_file}")
                dataframe = dataframe.map(
                    self.map_fn,
                    fn_kwargs={"data_source": "Maxwell-Jia/AIME_2024"},
                    remove_columns=cols
                )

            else:
                print(f"使用默认(旧版)逻辑处理: {parquet_file}")
                dataframe = dataframe.map(self.map_fn2, num_proc=16)

            dataframes.append(dataframe)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        print(f"Total dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict, *, data_source: str = None):
        # 处理 AIME 格式
        if data_source == "Maxwell-Jia/AIME_2024":
            problem, answer = row["Problem"], row["Answer"]
        else:  # yentinglin/aime_2025
            problem, answer = row["problem"], row["answer"]

        prompt = problem + answer_format

        return {
            "data_source": "aime_2025",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": "tool_agent",
        }

    def map_fn2(self, row: dict):
        content = row["prompt"][0]["content"]
        row["prompt"][0]["content"] = content + answer_format
        row["agent_name"] = "tool_agent"
        return row


def compute_score_pure_efficiency_0127(data_source, solution_str, ground_truth, extra_info, **kwargs):
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
    is_correct = result["score"] > 0.5

    num_turns = extra_info.get("num_turns", 0)
    MAX_TURNS = 32
    clipped_turns = min(num_turns, MAX_TURNS)
    if is_correct:
        base_score = 2.0
        penalty = clipped_turns * (1.0 / 32.0)
        final_score = base_score - penalty
    else:
        final_score = 0.0
    result["score"] = final_score
    if result["pred"] is None:
        result["pred"] = ""

    return result





