#!/usr/bin/env python3
"""
Simple test to verify PubMedBERT installation and basic functionality
"""

try:
    # Test imports
    print("Testing imports...")
    import torch
    from transformers import AutoTokenizer, AutoModel
    import numpy as np
    print("✅ All imports successful")

    # Test PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Test model loading
    print("\nLoading PubMedBERT model...")
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    print("✅ Model loaded successfully!")
    print(f"Model hidden size: {model.config.hidden_size}")
    print(f"Model vocab size: {model.config.vocab_size}")

    # Test basic encoding
    print("\nTesting basic encoding...")
    test_text = "This is a test of cancer research and immunotherapy treatments."

    # Tokenize
    inputs = tokenizer(test_text, return_tensors='pt', padding=True, truncation=True)
    print(f"Input tokens shape: {inputs['input_ids'].shape}")

    # Generate embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # Use CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :].numpy()

    print(f"✅ Embedding generated successfully!")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[0][:5]}")

    print(f"\n🎉 PubMedBERT is installed and working correctly!")
    print(f"You can now use the PubMedBERTEmbedder class.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing packages:")
    print("pip install torch transformers numpy")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your internet connection for model download.")