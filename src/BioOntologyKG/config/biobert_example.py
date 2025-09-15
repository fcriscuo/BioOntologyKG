from PubMedBERTEmbedder import PubMedBERTEmbedder

# Initialize embedder
embedder = PubMedBERTEmbedder()

# Generate embeddings
abstract_text = "This study investigates cancer immunotherapy..."
embedding = embedder.encode(abstract_text)

print(f"Embedding shape: {embedding.shape}")  # Should be (768,) for base model
print(f"First 10 values: {embedding[:10]}")