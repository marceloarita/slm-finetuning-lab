"""
Day 01 — Tokenization intro.

Key questions this script answers:
  1. How does a model "see" raw text?
  2. What is the relationship between words and tokens?
  3. How does tokenization differ across languages?
  4. What is the impact of sequence length on training?
"""

from transformers import AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


def load_tokenizer():
    print(f"Loading tokenizer: {MODEL_NAME}\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Vocabulary size: {tokenizer.vocab_size:,} tokens")
    print(f"Model max length: {tokenizer.model_max_length:,} tokens\n")
    return tokenizer


def explore_tokens(tokenizer, text: str, label: str = ""):
    """Show how a text is broken into tokens and mapped to IDs."""
    if label:
        print(f"--- {label} ---")
    print(f"Text:   {text!r}")

    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text, add_special_tokens=False)

    print(f"Tokens: {tokens}")
    print(f"IDs:    {ids}")
    print(f"Count:  {len(tokens)} tokens for {len(text.split())} words "
          f"({len(tokens) / len(text.split()):.2f} tokens/word)\n")


def explore_special_tokens(tokenizer):
    """Special tokens delimit sequences and signal start/end to the model."""
    print("--- Special tokens ---")
    text = "Hello, world!"
    ids_with = tokenizer.encode(text, add_special_tokens=True)
    ids_without = tokenizer.encode(text, add_special_tokens=False)
    print(f"Without special tokens: {ids_without}")
    print(f"With special tokens:    {ids_with}")
    print(f"Special tokens added:   {set(ids_with) - set(ids_without)}\n")


def explore_truncation(tokenizer):
    """Sequences longer than max_length get truncated during training."""
    print("--- Truncation and padding ---")
    long_text = "fine-tuning " * 200  # intentionally long
    full = tokenizer.encode(long_text, add_special_tokens=False)
    truncated = tokenizer.encode(
        long_text,
        add_special_tokens=False,
        max_length=128,
        truncation=True,
    )
    print(f"Original length:  {len(full)} tokens")
    print(f"Truncated length: {len(truncated)} tokens")
    print("→ Training always pads/truncates to a fixed max_length.\n")


def decode_round_trip(tokenizer, text: str):
    """Encoding → decoding should reconstruct the original text."""
    print("--- Round-trip: encode → decode ---")
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"Original: {text!r}")
    print(f"Decoded:  {decoded!r}")
    match = text.strip() == decoded.strip()
    print(f"Match: {'✓' if match else '✗'}\n")


def main():
    print("=" * 50)
    print("Day 01 — Tokenization Intro")
    print("=" * 50 + "\n")

    tokenizer = load_tokenizer()

    # 1. Basic tokenization
    explore_tokens(tokenizer, "Fine-tuning is powerful.", label="Simple sentence")

    # 2. Subword tokenization — uncommon words get split
    explore_tokens(tokenizer, "Backpropagation through time is tricky.", label="Technical words")

    # 3. Language comparison — same meaning, different token cost
    explore_tokens(tokenizer, "The cat sat on the mat.", label="English")
    explore_tokens(tokenizer, "O gato sentou no tapete.", label="Portuguese")

    # 4. Special tokens
    explore_special_tokens(tokenizer)

    # 5. Truncation
    explore_truncation(tokenizer)

    # 6. Round-trip
    decode_round_trip(tokenizer, "Fine-tuning small language models is fun!")

    print("=" * 50)
    print("Key takeaways:")
    print("  • Models read token IDs, not raw text")
    print("  • Subword tokenization handles unknown words gracefully")
    print("  • Non-English text is generally less token-efficient")
    print("  • max_length in training = max tokens, not max words")
    print("=" * 50)


if __name__ == "__main__":
    main()