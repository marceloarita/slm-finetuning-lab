"""
Day 01 — UTF-8 byte explorer.
Input any text and see exactly how Python encodes it into bytes.
"""


def explore(text: str):
    print(f"\nInput:  {text!r}")
    print("-" * 40)
    for char in text:
        byte_values = list(char.encode("utf-8"))
        n_bytes = len(byte_values)
        print(f"  {char!r:6} → {str(byte_values):20} ({n_bytes} byte{'s' if n_bytes > 1 else ' '})")
    total = list(text.encode("utf-8"))
    print("-" * 40)
    print(f"Full:   {total}")
    print(f"Total:  {len(total)} bytes for {len(text)} characters")


def main():
    print("UTF-8 Byte Explorer")
    print("Type anything and see its bytes. Ctrl+C to quit.\n")
    print("Try: A, ã, あ, 🔥, fine-tuning, こんにちは")

    while True:
        try:
            text = input("\n→ ")
            if text:
                explore(text)
        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()