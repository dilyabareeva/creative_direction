from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch

from datasets import load_dataset
from only_connect_dataset import (
    generate_task1_prompt,
    parse_oc_dataset,
    tokenize_oc_for_repr_ft,
    collate_fn_repr_ft,
    get_oc_train_test_val,
    gemma_messages,
    generate_task1_prompt_definite,
)

import pandas as pd

# %% md
OUTPUT_DIR = "./reports"
INTERVENTION_LAYER = 10
PYREFT_OUTPUT_DIR = f"{OUTPUT_DIR}/pyreft/"

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

DEVICE = "cuda:1"
remote = False
batch_size = 8


def evaluate_before():
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2-2b-it",
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it",
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
    )

    _, test_oc_dataset, val_oc_dataset = get_oc_train_test_val()
    print(generate_task1_prompt(test_oc_dataset[0]["group"]))
    batched_dataset = test_oc_dataset.shuffle(27).take(100).batch(batch_size)
    results = []
    for batch in batched_dataset:
        for i in range(len(batch["group"])):
            group = batch["group"][i]
            gt_connection = batch["gt_connection"][i]
            prompt = generate_task1_prompt_definite(group)
            messages = gemma_messages(prompt)
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(DEVICE)
            input_len = input_ids.shape[1]
            output_ids = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
            )
            new_tokens = output_ids[0][input_len:]
            model_output = str(tokenizer.decode(new_tokens, skip_special_tokens=True))
            results.append(
                {
                    "group": group,
                    "gt_connection": gt_connection,
                    "model_output": model_output.split("\n")[0],
                }
            )
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    df.to_csv(f"{OUTPUT_DIR}/gemma_2_2b_it_before.csv", index=False, sep="\t")


if __name__ == "__main__":
    evaluate_before()
