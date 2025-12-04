"""
Script to download and run the moonshotai/Kimi-K2-Thinking model
This will use the default cache location (~/.cache/huggingface/)
uv add tiktoken
uv add compressed-tensors
uv  add encodings
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys


def main():
    print("=" * 60)
    print("Kimi-K2-Thinking Model Runner")
    print("=" * 60)

    model_name = "moonshotai/Kimi-K2-Thinking"
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if device == "cpu":
        print("WARNING: Running on CPU. This will be slow for large models.")
        print("Consider using a system with GPU for better performance.\n")

    try:
        # Load tokenizer
        print(f"\nLoading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded successfully")

        # Load model
        print(f"\nLoading model from {model_name}...")
        print("(This may take a while on first run as it downloads the model)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

        if device == "cpu":
            model = model.to(device)

        print("✓ Model loaded successfully")

        # Example prompt - adjust based on model's expected format
        prompt = "What is the sum of all prime numbers between 1 and 20? Please show your reasoning step by step."

        print("\n" + "=" * 60)
        print("PROMPT:")
        print("=" * 60)
        print(prompt)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate response
        print("\n" + "=" * 60)
        print("GENERATING RESPONSE...")
        print("=" * 60)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("\n" + "=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(response)
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n❌ Error occurred: {type(e).__name__}")
        print(f"Details: {str(e)}")
        print("\nTroubleshooting tips:")
        print("- Ensure you have enough disk space for the model")
        print("- Check your internet connection")
        print("- Verify you have sufficient RAM/VRAM")
        print("- You may need to accept the model's license on HuggingFace")
        sys.exit(1)


if __name__ == "__main__":
    print("\nRequired packages: transformers, torch, accelerate")
    print("Install with: pip install transformers torch accelerate\n")
    main()