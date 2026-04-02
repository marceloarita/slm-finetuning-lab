"""
Day 01 — BPE (Byte-Pair Encoding) implemented from scratch.
No libraries — pure Python only.

Steps:
  1. Build initial vocabulary from characters
  2. Count most frequent pairs
  3. Merge and repeat
  4. Watch the vocabulary grow
"""

from collections import Counter


# ── Corpus ────────────────────────────────────────────────────────────────────

CORPUS = [
    "gato gato gato",
    "rato rato",
    "gato rato",
    "pato pato pato pato",
    "gato pato",
]


# ── Step 1: represent each word as a sequence of characters + end marker ──────

def build_vocab(corpus: list[str]) -> dict[tuple, int]:
    """
    Each word becomes a tuple of characters.
    </s> marks the end of a word — helps the tokenizer learn word boundaries.
    Returns a dict of {word_as_tuple: frequency}.
    """
    vocab = Counter()
    for sentence in corpus:
        for word in sentence.split():
            # 'gato' → ('g', 'a', 't', 'o', '</s>')
            chars = tuple(list(word) + ["</s>"])
            vocab[chars] += 1
    return dict(vocab)


# ── Step 2: count all adjacent pairs ─────────────────────────────────────────

def count_pairs(vocab: dict[tuple, int]) -> Counter:
    """Count how often each adjacent pair appears across all words."""
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


# ── Step 3: merge the most frequent pair ─────────────────────────────────────

def merge_pair(pair: tuple[str, str], vocab: dict[tuple, int]) -> dict[tuple, int]:
    """Replace every occurrence of `pair` in vocab with the merged token."""
    merged = pair[0] + pair[1]
    new_vocab = {}
    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


# ── Step 4: tokenize a new word using learned merges ─────────────────────────

def tokenize(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Apply learned merges to tokenize a new word."""
    tokens = list(word) + ["</s>"]
    for pair in merges:
        merged = pair[0] + pair[1]
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("BPE From Scratch")
    print("=" * 55)

    # Build initial vocab
    vocab = build_vocab(CORPUS)

    print("\nCorpus:")
    for s in CORPUS:
        print(f"  {s!r}")

    print("\nInitial vocabulary (word → frequency):")
    for word, freq in vocab.items():
        print(f"  {' '.join(word):30} → {freq}x")

    # Track all merges
    merges = []
    n_merges = 8

    print(f"\nRunning {n_merges} merges...\n")

    for i in range(n_merges):
        pairs = count_pairs(vocab)
        if not pairs:
            break

        # Find most frequent pair
        best_pair = max(pairs, key=pairs.get)
        best_freq = pairs[best_pair]

        # Merge it
        vocab = merge_pair(best_pair, vocab)
        merges.append(best_pair)

        merged_token = best_pair[0] + best_pair[1]
        print(f"Merge {i+1:2}: {best_pair[0]!r:8} + {best_pair[1]!r:8} → {merged_token!r:12} (freq: {best_freq})")

        # Show current vocab state
        print(f"         Vocab state:")
        for word, freq in vocab.items():
            print(f"           {' '.join(word):30} → {freq}x")
        print()

    # Final tokenization examples
    print("=" * 55)
    print("Tokenizing new words with learned merges:")
    test_words = ["gato", "rato", "pato", "gatorato", "mato"]
    for word in test_words:
        tokens = tokenize(word, merges)
        print(f"  {word:12} → {tokens}")

    print("=" * 55)
    print(f"Final merge rules learned: {merges}")
    print("=" * 55)
    return merges


def interactive(merges: list[tuple[str, str]]):
    print("\nTokenizer interactive mode.")
    print("Type any word to see its tokens. Ctrl+C to quit.\n")
    while True:
        try:
            word = input("→ ").strip()
            if word:
                tokens = tokenize(word, merges)
                print(f"  tokens : {tokens}")
                print(f"  count  : {len(tokens)}\n")
        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    merges = main()
    interactive(merges)