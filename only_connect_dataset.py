import random
import torch
from datasets import load_dataset

EXAMPLES = (
    "Bloody, Virgin, Hail, Typhoid – Connection: Mary\n"
    "Avid, Uke, Liver, Ames – Connection: Boys names missing a "
    "letter"
)


def gemma_messages(content):
    return [
        {"role": "user", "content": content},
    ]


def qwen_cot_messages(content):
    return [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": content},
    ]


PROMPT_TEMPLATES = [
    (
        'Your task: given 4 "clues" (words or phrases), determine the common connection among them. '
        "You will be given the clues as a list. You are also provided with examples of solved connections, which include the common link. "
        "Provide your answer as a single word or phrase that precisely identifies the specific connection that all the clues share, without quotes or any special characters.\n"
        "Do not invent connections not supported by the clues or provide an overly general answer.\n"
        "Note: Connections might be thematic, linguistic, factual, or mathematical and can rely on both arcane subject areas and popular culture.\n\n"
        "Example solved connection(s):\n{examples}\n\n"
        "Clues: {clues}\n\n"
    ),
    (
        "Your goal: Given a set of four 'clues' (words or phrases), identify the single connection that links them. "
        "Clues will be presented as a list, and you will also receive examples of correctly solved connections, including their common link. "
        "Respond with a precise single word or phrase that accurately captures the specific relationship among the clues, without quotes or any special characters.\n"
        "Do not generate unsupported connections or give overly broad answers.\n"
        "Connections may be thematic, linguistic, factual, or mathematical, drawing from niche topics and popular culture alike.\n\n"
        "Example solved connections:\n{examples}\n\n"
        "Clues: {clues}\n\n"
    ),
    (
        "Your objective is to analyze a set of four 'clues' (words or phrases) and determine the specific connection that links them. "
        "The clues will be provided in list form, along with a set of example solved connections that explicitly state their common link. "
        "Your response must be a single word or phrase that accurately captures the precise connection shared by all four clues, without quotes or any special characters."
        "Avoid making assumptions beyond what is explicitly supported by the clues, and refrain from giving overly broad or vague answers.\n"
        "Connections may be derived from thematic, linguistic, factual, or mathematical relationships and can encompass both obscure trivia and well-known cultural references.\n\n"
        "Example solved connection(s):\n{examples}\n\n"
        "Clues: {clues}\n\n"
    ),
    (
        "Your challenge is to determine the common link among four 'clues' (words or phrases) that will be presented as a list. "
        "To assist you, examples of previously solved connections—including their correct common links—are provided. "
        "Your answer must be a single word or phrase that precisely and specifically captures the connection among all four clues, without quotes or any special characters."
        "Do not introduce connections that are not explicitly supported by the clues, and avoid overly broad or generic answers.\n"
        "Connections may be thematic, linguistic, factual, or mathematical in nature and could draw from both obscure subject areas and mainstream culture.\n\n"
        "Example solved connection(s):\n{examples}\n\n"
        "Clues: {clues}\n\n"
    ),
]


def generate_task1_prompt(clues):
    """
    Generate a randomly selected prompt from different styles, inserting clues and examples.

    Parameters:
        clues (list): A list of clue strings.
        examples (str): A string containing example solved connections.

    Returns:
        str: A single, randomly chosen formatted prompt.
    """
    joined_clues = ", ".join(clues)
    template = random.choice(PROMPT_TEMPLATES)
    return template.format(clues=joined_clues, examples=EXAMPLES)


def generate_task1_prompt_definite(clues):
    """
    Generate a randomly selected prompt from different styles, inserting clues and examples.

    Parameters:
        clues (list): A list of clue strings.
        examples (str): A string containing example solved connections.

    Returns:
        str: A single, randomly chosen formatted prompt.
    """
    joined_clues = ", ".join(clues)
    template = PROMPT_TEMPLATES[0]
    return template.format(clues=joined_clues, examples=EXAMPLES)


def parse_oc_dataset(batch):
    groups = []
    connections = []
    for i in range(len(batch["wall_id"])):
        for j in range(1, 5):
            group_key = f"group_{j}"
            group_data = batch[group_key][i]
            groups.append(group_data["gt_words"])
            connections.append(group_data["gt_connection"])
    return {"group": groups, "gt_connection": connections}


def tokenize_oc_for_repr_ft(sample, tokenizer, template_fn):
    prompt = generate_task1_prompt(sample["group"])
    messages = gemma_messages(prompt)
    input_id = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if "_" in sample["gt_connection"]:
        sample["gt_connection"] = (
            sample["gt_connection"].replace("_", "").strip().title()
        )
    sample["gt_connection"] = (
        sample["gt_connection"][0].upper() + sample["gt_connection"][1:]
    )
    output_id = tokenizer(sample["gt_connection"]).input_ids[1:]
    seq_id = input_id + output_id
    intervention_id = [0] * len(seq_id)
    intervention_id[len(input_id) : len(input_id) + len(output_id)] = [1] * len(
        output_id
    )
    return {"input_ids": seq_id, "intervention_ids": intervention_id}


def collate_fn_repr_ft(batch):
    max_length = max([len(item["input_ids"]) for item in batch])
    collated = {}
    # Collate these keys into tensors
    collated["input_ids"] = torch.zeros((len(batch), max_length), dtype=torch.long)
    collated["intervention_ids"] = torch.zeros(
        (len(batch), max_length), dtype=torch.bool
    )

    for i, item in enumerate(batch):
        collated["input_ids"][i, : len(item["input_ids"])] = torch.tensor(
            item["input_ids"]
        )
        collated["intervention_ids"][i, : len(item["intervention_ids"])] = torch.tensor(
            item["intervention_ids"]
        )

    return collated


def get_oc_train_test_val():
    dataset = load_dataset("TaatiTeam/OCW")["test"]

    train_oc_dataset = dataset.map(
        parse_oc_dataset,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )

    test_dataset = load_dataset("TaatiTeam/OCW")["train"]
    test_oc_dataset = test_dataset.map(
        parse_oc_dataset,
        batched=True,
        remove_columns=test_dataset.column_names,
        load_from_cache_file=False,
    )
    split_dataset = test_oc_dataset.shuffle(seed=42).train_test_split(
        test_size=0.1, seed=42
    )
    # Access train and test splits
    test_oc_dataset = split_dataset["train"]
    val_oc_dataset = split_dataset["test"]
    return train_oc_dataset, test_oc_dataset, val_oc_dataset
