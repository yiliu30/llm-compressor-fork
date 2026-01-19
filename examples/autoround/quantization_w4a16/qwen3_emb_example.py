from auto_round.calib_dataset import get_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier
from llmcompressor.utils import dispatch_for_generation
import torch
from torch import Tensor
import torch.nn.functional as F

# Select model and load it.
# model_id = "Qwen/Qwen3-235B-A22B"
model_id = "Qwen/Qwen3-Embedding-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
model = AutoModel.from_pretrained(model_id, dtype="auto")


def verify_model(model: torch.nn.Module, msg: str | None = ""):
    with torch.no_grad():

        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f"Instruct: {task_description}\nQuery:{query}"

        # Each query must come with a one-sentence instruction that describes the task
        task = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

        queries = [
            get_detailed_instruct(task, "What is the capital of China?"),
            get_detailed_instruct(task, "Explain gravity"),
        ]
        # No need to add instruction for retrieval documents
        documents = [
            "The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
        ]
        input_texts = queries + documents

        # We recommend enabling flash_attention_2 for better acceleration and memory saving.
        # model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B', attn_implementation="flash_attention_2", torch_dtype=torch.float16).cuda()

        max_length = 8192

        # Tokenize the input texts
        batch_dict = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict.to(model.device)
        outputs = model(**batch_dict)

        def last_token_pool(
            last_hidden_states: Tensor, attention_mask: Tensor
        ) -> Tensor:
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device),
                    sequence_lengths,
                ]

        embeddings = last_token_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = embeddings[:2] @ embeddings[2:].T
        print(f"{msg} similarity scores: {scores.tolist()}")


verify_model(model, "original model")

# Select calibration dataset.
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048
ITERS = 200
# Get aligned calibration dataset.

ds = get_dataset(
    tokenizer=tokenizer,
    seqlen=MAX_SEQUENCE_LENGTH,
    nsamples=NUM_CALIBRATION_SAMPLES,
)



recipe = AutoRoundModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "lm_head",
        "re:.*mlp.gate$",
    ],
    iters=ITERS,
    enable_torch_compile=False,
)


# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    shuffle_calibration_samples=False,
)

verify_model(model, "after quantization")

"""
original model similarity scores: [[0.75390625, 0.11279296875], [0.0303955078125, 0.62109375]]
after quantization similarity scores: [[0.765625, 0.12158203125], [0.039794921875, 0.6171875]]
"""