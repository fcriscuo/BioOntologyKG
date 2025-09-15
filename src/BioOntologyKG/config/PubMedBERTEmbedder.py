#!/usr/bin/env python3
"""
Direct usage of PubMedBERT for generating embeddings
No separate module import needed - everything in one file
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging
from typing import List, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PubMedBERTEmbedder:
    """PubMedBERT embedder using transformers library directly"""

    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                 cache_dir: str = None, device: str = None):
        """
        Initialize PubMedBERT model

        Args:
            model_name: HuggingFace model identifier
            cache_dir: Local directory to cache model files
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "huggingface" / "transformers")

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading PubMedBERT model: {model_name}")

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=False
            )

            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=False
            )

            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def encode(self, texts: Union[str, List[str]], max_length: int = 512,
               batch_size: int = 8, pooling_strategy: str = 'cls') -> np.ndarray:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text string or list of texts
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            pooling_strategy: 'cls', 'mean', or 'max'

        Returns:
            numpy array of embeddings
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._encode_batch(batch_texts, max_length, pooling_strategy)
            all_embeddings.append(batch_embeddings)

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)

        # Return single embedding if input was single text
        if len(texts) == 1:
            return embeddings[0]

        return embeddings

    def _encode_batch(self, texts: List[str], max_length: int, pooling_strategy: str) -> np.ndarray:
        """Encode a batch of texts"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Get embeddings based on pooling strategy
            if pooling_strategy == 'cls':
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif pooling_strategy == 'mean':
                # Mean pooling over all tokens (excluding padding)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                    input_mask_expanded.sum(1), min=1e-9)
            elif pooling_strategy == 'max':
                # Max pooling over all tokens
                embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        return embeddings.cpu().numpy()

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        return dot_product / (norm1 * norm2)


def main():
    """Basic usage example"""
    print("=" * 60)
    print("PubMedBERT Basic Usage Example")
    print("=" * 60)

    try:
        # Initialize embedder
        print("Initializing PubMedBERT embedder...")
        embedder = PubMedBERTEmbedder()

        # Example abstract text
        abstract_text = """
        This study investigates the molecular mechanisms underlying cancer cell 
        proliferation and identifies potential therapeutic targets for treatment. 
        We used CRISPR-Cas9 gene editing to knock out specific oncogenes and 
        observed significant reduction in tumor growth in mouse models.
        """

        print(f"Generating embedding for abstract...")
        print(f"Abstract: {abstract_text.strip()[:100]}...")

        # Generate embedding
        embedding = embedder.encode(abstract_text.strip())

        print(f"\n✅ Success!")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 10 values: {embedding[:10]}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")

        # Test with multiple abstracts
        print(f"\nTesting with multiple abstracts...")
        abstracts = [
            "Cancer immunotherapy has shown promising results in clinical trials.",
            "Machine learning algorithms can accelerate drug discovery processes.",
            "CRISPR gene editing enables precise DNA modifications in living cells."
        ]

        embeddings = embedder.encode(abstracts, batch_size=2)
        print(f"Generated embeddings shape: {embeddings.shape}")

        # Test similarity
        print(f"\nTesting similarity between abstracts...")
        for i, abstract in enumerate(abstracts):
            similarity = embedder.similarity(abstract_text.strip(), abstract)
            print(f"Similarity with abstract {i + 1}: {similarity:.4f}")
            print(f"  '{abstract[:50]}...'")

        print(f"\n🎉 All tests passed! PubMedBERT is working correctly.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Make sure you have installed: pip install torch transformers numpy")
        raise


if __name__ == "__main__":
    main()