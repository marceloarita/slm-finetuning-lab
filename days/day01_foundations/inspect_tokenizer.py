"""
Day 01 — Inspect the real merge rules of SmolLM2's tokenizer.
Downloads tokenizer.json from Hugging Face and explores its internals.
"""

from transformers import AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def inspect_vocabulary(tokenizer):
    vocab = tokenizer.get_vocab()
    print(f"Total tokens in vocabulary: {len(vocab):,}")

    print("\nFirst 20 tokens (by ID):")
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    for token, idx in sorted_vocab[:20]:
        print(f"  ID {idx:5} → {token!r}")

    print("\nLast 10 tokens (by ID):")
    for token, idx in sorted_vocab[-10:]:
        print(f"  ID {idx:5} → {token!r}")


def inspect_merges(tokenizer):
    # Load raw merge rules directly from tokenizer.json
    import json
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=MODEL_NAME, filename="tokenizer.json")
    with open(path) as f:
        data = json.load(f)

    merges = data["model"]["merges"]
    total = len(merges)
    print(f"\nTotal merge rules: {total:,}")

    print("\nFirst 20 merges (learned first = most frequent in training corpus):")
    for i, merge in enumerate(merges[:20]):
        a, b = merge.split(" ")
        print(f"  Merge {i+1:5}: {a!r:15} + {b!r:15} → {a+b!r}")

    print("\nLast 5 merges (learned last = least frequent):")
    for i, merge in enumerate(merges[-5:]):
        idx = total - 5 + i
        a, b = merge.split(" ")
        print(f"  Merge {idx:5}: {a!r:15} + {b!r:15} → {a+b!r}")


def inspect_special_tokens(tokenizer):
    print("\nSpecial tokens:")
    for name, token in tokenizer.special_tokens_map.items():
        idx = tokenizer.convert_tokens_to_ids(token)
        print(f"  {name:20} → {token!r:15} (ID: {idx})")


def search_token(tokenizer, query: str):
    vocab = tokenizer.get_vocab()
    print(f"\nSearching for tokens containing {query!r}:")
    matches = [(t, i) for t, i in vocab.items() if query.lower() in t.lower()]
    matches.sort(key=lambda x: x[1])
    if matches:
        for token, idx in matches[:10]:
            print(f"  ID {idx:5} → {token!r}")
        if len(matches) > 10:
            print(f"  ... and {len(matches) - 10} more")
    else:
        print("  No matches found.")


def main():
    print("=" * 55)
    print("SmolLM2 Tokenizer Inspector")
    print("=" * 55)

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\n--- Vocabulary ---")
    inspect_vocabulary(tokenizer)

    print("\n--- Merge Rules ---")
    inspect_merges(tokenizer)

    print("\n--- Special Tokens ---")
    inspect_special_tokens(tokenizer)

    print("\n--- Token Search ---")
    search_token(tokenizer, "fine")
    search_token(tokenizer, "tun")

    print("\n" + "=" * 55)
    print("Interactive token search. Ctrl+C to quit.")
    print("=" * 55)
    while True:
        try:
            query = input("\nSearch token → ").strip()
            if query:
                search_token(tokenizer, query)
        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()