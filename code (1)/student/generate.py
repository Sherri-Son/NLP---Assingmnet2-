import torch

from data_utils import load_memmap_dataset, load_vocab, decode
from student.model import TransformerLM, softmax


@torch.no_grad()
def generate(
    model,
    prompt_ids,
    max_new_tokens=100,
    temperature=1.0,
    top_p=0.9,
    eos_token_id=None,
):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")

    device = next(model.parameters()).device
    ids = torch.as_tensor(
        prompt_ids, dtype=torch.long, device=device
    )

    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("Provide a nonempty, one-dimensional prompt")

    ids = ids.unsqueeze(0)
    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            context = ids[:, -model.context_length:]

            logits = model(context)[0, -1, :]
            probs = softmax(logits / temperature, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_ids = torch.sort(
                    probs, descending=True
                )
                cumulative = torch.cumsum(sorted_probs, dim=-1)

                remove = cumulative > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False

                filtered = sorted_probs.masked_fill(remove, 0.0)
                filtered = filtered / filtered.sum()

                sampled_index = torch.multinomial(filtered, 1)
                next_token = sorted_ids[sampled_index]
            else:
                next_token = torch.multinomial(probs, 1)

            ids = torch.cat([ids, next_token.reshape(1, 1)], dim=1)

            if (
                eos_token_id is not None
                and next_token.item() == eos_token_id
            ):
                break
    finally:
        model.train(was_training)

    return ids[0].tolist()


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        "outputs/model.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = TransformerLM(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    vocab, _ = load_vocab("data/tinystories_vocab_merges.pt")

    eos_token_id = next(
        (token_id for token_id, value in vocab.items()
         if value == b"<|endoftext|>"),
        None,
    )
    if eos_token_id is None:
        raise ValueError("Could not find the end-of-text token")

    dataset = load_memmap_dataset(
        "data/tinystories_valid.bin",
        "data/tinystories_valid.meta",
    )

    prompt_ids = dataset[:12].tolist()

    output_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9,
        eos_token_id=eos_token_id,
    )

    print("PROMPT:")
    print(decode(prompt_ids, vocab))

    print("\nGENERATED CONTINUATION:")
    print(decode(output_ids[len(prompt_ids):], vocab))


if __name__ == "__main__":
    main()