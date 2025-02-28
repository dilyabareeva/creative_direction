import argparse

from nnsight import LanguageModel
from transformers import AutoTokenizer
import os
import torch

from datasets import load_dataset

from interventions import OneDAffineTransformation, AddVector
from only_connect_dataset import (
    get_oc_train_test_val,
    gemma_messages,
    generate_task1_prompt_definite,
)

import pandas as pd

OUTPUT_DIR = "./reports"
INTERVENTION_LAYER = 10

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

remote = False
BATCH_SIZE = 8
DEVICE = "cuda:1"

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-2b-it",
    model_max_length=1024,
    padding_side="right",
    use_fast=False,
)
model = LanguageModel(
    "google/gemma-2-2b-it",
    device_map=DEVICE,
    torch_dtype=torch.bfloat16,
    dispatch=True,
)

_, test_oc_dataset, val_oc_dataset = get_oc_train_test_val()


def evaluate_after(p_type, alpha):
    """Generates the output of the model after the intervention.

    Args:
        p_type (str): The type of intervention to be applied, either "add",
         to add a constant steering vector or "affine", for an affine
         transformation.

    """

    postfix = f"{p_type}_alpha_{alpha}"
    save_path = f"./out/subspace_proj_{postfix}.pt"
    if p_type.lower() == "add":
        subspace_proj = AddVector(embed_dim=model.config.hidden_size).to(DEVICE)
        subspace_proj_dict = torch.load(save_path)
        subspace_proj.v = torch.nn.Parameter(subspace_proj_dict["v"])
    elif p_type.lower() == "affine":
        subspace_proj = OneDAffineTransformation(embed_dim=model.config.hidden_size)
        subspace_proj_dict = torch.load(save_path)
        subspace_proj.beta = torch.nn.Parameter(subspace_proj_dict["alpha"])
        subspace_proj.v = torch.nn.Parameter(subspace_proj_dict["v"])
        subspace_proj.to(DEVICE)
    batched_dataset = test_oc_dataset.shuffle(27).take(100).batch(BATCH_SIZE)
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
            batch_size = 1
            all_inds = torch.arange(batch_size)

            with model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
            ) as tracer:
                base = model.model.layers[INTERVENTION_LAYER].output[0].save()

                B = base[all_inds, :, :]
                n_pos = B.shape[1]
                mixed_out = subspace_proj(B.view(batch_size * n_pos, -1), batch_size)
                mixed_out = mixed_out.view(batch_size, n_pos, -1)
                model.model.layers[INTERVENTION_LAYER].output[0][all_inds, :, :] = (
                    mixed_out
                )
                modified_tokens = model.generator.output.save()

            new_tokens = modified_tokens[0][input_len:]
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
    df.to_csv(f"{OUTPUT_DIR}/gemma_2_2b_it_after_{postfix}.csv", index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate after with specified parameters."
    )
    parser.add_argument(
        "--p_type", type=str, required=True, help="Parameter type (string)"
    )
    parser.add_argument(
        "--alpha", type=float, required=True, help="Alpha value (float)"
    )

    args = parser.parse_args()

    p_type = args.p_type
    alpha = args.alpha

    evaluate_after(p_type, alpha)
