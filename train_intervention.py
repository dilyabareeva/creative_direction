import argparse
import gc
from functools import partial

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens.utils import tokenize_and_concatenate
from transformers import AutoTokenizer
from tqdm import trange, tqdm
import torch
import wandb
from nnsight import LanguageModel
from transformers import get_linear_schedule_with_warmup

from interventions import OneDAffineTransformation, AddVector
from only_connect_dataset import (
    tokenize_oc_for_repr_ft,
    collate_fn_repr_ft,
    get_oc_train_test_val,
    gemma_messages,
)

DEVICE = "cuda:1"
remote = False
N_EPOCHS = 1
INTERVENTION_LAYER = 10


def free_unused_device_memory():
    """Free unused cuda memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        raise RuntimeError("not using cuda")
    gc.collect()


def calculate_ft_loss(logits, labels, intervention_ids, vocab_size=32001):
    shift_logits = logits[..., :, :].contiguous()[intervention_ids]
    shift_labels = labels[..., :].contiguous()[intervention_ids]
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    return loss


# %%
def batch_intervention(inputs, INTERVENTION_LAYER, model: LanguageModel, subspace_proj):
    """
    Batched subspace_ablation intervention at a single layer using nnsight
    """
    batch_size, seq_length = inputs["intervention_ids"].shape
    all_inds = torch.arange(batch_size)
    base_prompt = inputs["input_ids"][:batch_size]

    with model.trace(validate=False, remote=remote) as tracer:
        with tracer.invoke(base_prompt, scan=False):
            base = model.model.layers[INTERVENTION_LAYER].output[0].save()

    with model.trace(validate=False, remote=remote) as tracer:
        # intervention
        with tracer.invoke(base_prompt, scan=False):
            B = base[all_inds, :, :]
            n_pos = B.shape[1]

            mixed_out = subspace_proj(B.view(batch_size * n_pos, -1), batch_size)
            mixed_out = mixed_out.view(batch_size, n_pos, -1)
            model.model.layers[INTERVENTION_LAYER].output[0][all_inds, :, :] = mixed_out

        save_out = model.output.save()

    del base_prompt, B
    free_unused_device_memory()

    output_logits = save_out.value.logits
    del save_out
    return output_logits


def batch_mse_loss(inputs, INTERVENTION_LAYER, model: LanguageModel, subspace_proj):
    """
    Batched subspace_ablation intervention at a single layer using nnsight
    """

    batch_size, seq_length = inputs["tokens"].shape
    all_inds = torch.arange(batch_size)
    base_prompt = inputs["tokens"][:batch_size]

    with model.trace(validate=False, remote=remote) as tracer:
        with tracer.invoke(base_prompt, scan=False):
            base = model.model.layers[INTERVENTION_LAYER].output[0].save()
        base_save_out = model.output.save()

    with model.trace(validate=False, remote=remote) as tracer:
        # intervention
        with tracer.invoke(base_prompt, scan=False):
            B = base[all_inds, :, :]
            n_pos = B.shape[1]

            mixed_out = subspace_proj(B.view(batch_size * n_pos, -1), batch_size)
            mixed_out = mixed_out.view(batch_size, n_pos, -1)
            model.model.layers[INTERVENTION_LAYER].output[0][all_inds, :, :] = mixed_out

        new_save_out = model.output.save()

    mse_loss = torch.nn.functional.mse_loss(
        base_save_out.value.logits, new_save_out.value.logits
    )

    del base_prompt, B
    free_unused_device_memory()
    del base_save_out, new_save_out
    return mse_loss


def train_intervention(p_type: str, alpha: float):
    """Use representation finetuning to learn a subspace projection.

    Args:
        p_type (str): Type of projection to use. Either "add", for adding
         a constant steering vector, or "affine", for an affine transformation.

    """

    postfix = f"{p_type}_alpha_{alpha}"
    save_path = f"./out/subspace_proj_only_connect_{postfix}.pt"
    # init wandb
    wandb.init(project="subspace-projection", name=f"{postfix}")

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
    tokenizer.pad_token_id = tokenizer.eos_token_id

    batch_size = 2  # more doesn't fit in memory

    # Load datasets
    train_oc_dataset, _, val_oc_dataset = get_oc_train_test_val()
    map_fn = partial(
        tokenize_oc_for_repr_ft, tokenizer=tokenizer, template_fn=gemma_messages
    )
    train_dataset = train_oc_dataset.map(
        map_fn,
        remove_columns=train_oc_dataset.column_names,
        load_from_cache_file=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_repr_ft,
    )
    mse_dataset = load_dataset(
        path="NeelNanda/pile-10k",
        split="train",
        streaming=False,
    )
    mse_token_dataset = tokenize_and_concatenate(
        dataset=mse_dataset,  # type: ignore
        tokenizer=model.tokenizer,  # type: ignore
        streaming=True,
        max_length=128,
        add_bos_token=False,
    ).shuffle(seed=42)
    mse_dataloader = DataLoader(
        mse_token_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    val_dataset = val_oc_dataset.map(
        map_fn, remove_columns=val_oc_dataset.column_names, load_from_cache_file=False
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_repr_ft
    )
    # Sanity check the model and prompts
    with model.generate(
        torch.tensor(train_dataset[24]["input_ids"]).unsqueeze(0).to(DEVICE),
        max_new_tokens=10,
        do_sample=False,
    ) as tracer:
        output = model.generator.output.save()
    print(tokenizer.decode(output[0], skip_special_tokens=False))

    if p_type.lower() == "add":
        subspace_proj = AddVector(embed_dim=model.config.hidden_size).to(DEVICE)
    elif p_type.lower() == "affine":
        subspace_proj = OneDAffineTransformation(embed_dim=model.config.hidden_size).to(
            DEVICE
        )
    else:
        raise ValueError(f"Unknown proj type: {p_type}")

    # Training (hyperparameters copied from nnsight Boundless DAS tutorial)
    gradient_accumulation_steps = 4

    t_total = int(len(train_dataloader) * N_EPOCHS)
    warm_up_steps = 0.1 * t_total
    # Define params to be learned
    optimizer_params = []
    for param in subspace_proj.parameters():
        optimizer_params.append({"params": param})
    optimizer = torch.optim.Adam(
        optimizer_params,
        lr=1e-1,  # for full_train_1_no_pres, used 1e-1
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
    )

    train_iterator = trange(0, N_EPOCHS, desc="Epoch")
    total_step = 0
    val_epoch_losses = []
    increasing_epochs_count = 0
    last_val_loss = float("inf")
    subspace_proj.train()

    for epoch in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
        )
        val_iterator = tqdm(
            val_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
        )
        mse_iterator = iter(mse_dataloader)

        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(DEVICE)

            transformed_outputs = batch_intervention(
                inputs, INTERVENTION_LAYER, model, subspace_proj
            )

            ce_loss = calculate_ft_loss(
                transformed_outputs[:, :-1, :],
                inputs["input_ids"][:, 1:],
                inputs["intervention_ids"][:, 1:],
                vocab_size=model.config.vocab_size,
            )

            if alpha == 0:
                preservation_loss = torch.tensor(0.0).to(DEVICE)
            else:
                preservation_loss = batch_mse_loss(
                    next(mse_iterator), INTERVENTION_LAYER, model, subspace_proj
                )
            loss = ce_loss + alpha * preservation_loss

            log_dict = {
                "loss": round(loss.item(), 2),
                "reft_loss": round(ce_loss.item(), 2),
                "preservation_loss": round(preservation_loss.item(), 2),
            }
            epoch_iterator.set_postfix(log_dict)
            wandb.log({**log_dict, "step": total_step})

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            if total_step % gradient_accumulation_steps == 0:
                if not (gradient_accumulation_steps > 1 and total_step == 0):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            total_step += 1

        # Validation epoch
        subspace_proj.eval()
        val_epoch_loss = 0.0
        val_steps_count = 0

        with torch.no_grad():
            for step, inputs in enumerate(val_iterator):
                for k, v in inputs.items():
                    if v is not None and isinstance(v, torch.Tensor):
                        inputs[k] = v.to(DEVICE)

                transformed_outputs = batch_intervention(
                    inputs, INTERVENTION_LAYER, model, subspace_proj
                )

                loss = calculate_ft_loss(
                    transformed_outputs[:, :-1, :],
                    inputs["input_ids"][:, 1:],
                    inputs["intervention_ids"][:, 1:],
                    vocab_size=model.config.vocab_size,
                )

                log_dict = {
                    "val_reft_loss": round(loss.item(), 2),
                }
                epoch_iterator.set_postfix(log_dict)

                val_epoch_loss += loss.item()
                val_steps_count += 1

        val_epoch_losses.append(val_epoch_loss / val_steps_count)

        print(f"Epoch {epoch} - Val Total: {val_epoch_losses[-1]:.4f}")
        wandb.log({"val_loss": val_epoch_losses[-1]})

        # Early stopping check
        current_val_loss = val_epoch_losses[-1]
        if current_val_loss > last_val_loss:
            increasing_epochs_count += 1
        else:
            increasing_epochs_count = 0

        if increasing_epochs_count >= 2:
            print(
                f"Validation loss has increased for {increasing_epochs_count} consecutive epochs. Stopping training."
            )
            break

        last_val_loss = current_val_loss
        subspace_proj.train()
    save_path = f"./out/subspace_proj_{postfix}.pt"
    if p_type.lower() == "add":
        save_param_dict = {
            "v": subspace_proj.v.detach().cpu(),
        }
    elif p_type.lower() == "affine":
        save_param_dict = {
            "v": subspace_proj.v.detach().cpu(),
            "alpha": subspace_proj.beta.detach().cpu(),
        }
    torch.save(save_param_dict, save_path)
    wandb.finish()
    print(f"Saved projection parameters to {save_path}")


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

    train_intervention(p_type, alpha)
