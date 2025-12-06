#!/usr/bin/env python3
"""
Script to calculate and persist embeddings for PubMedArticle nodes.
Processes abstracts from ReadCube Papers library that lack embeddings.
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


class PubMedEmbeddingProcessor:
    """Process and persist embeddings for PubMedArticle abstracts."""

    def __init__(self):
        """Initialize processor with Neo4j connection and embedding model."""
        # Get Neo4j credentials from environment
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_username = os.getenv('NEO4J_USERNAME')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        self.neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')

        # Get embedding model name
        self.embedding_model_name = os.getenv('PUBMED_EMBEDDING_MODEL')

        # Validate environment variables
        self._validate_config()

        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )

        # Initialize embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        print("Embedding model loaded successfully")

    def _validate_config(self):
        """Validate required environment variables are set."""
        required_vars = {
            'NEO4J_URI': self.neo4j_uri,
            'NEO4J_USERNAME': self.neo4j_username,
            'NEO4J_PASSWORD': self.neo4j_password,
            'PUBMED_EMBEDDING_MODEL': self.embedding_model_name
        }

        missing_vars = [var for var, val in required_vars.items() if not val]

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    def get_articles_needing_embeddings(self) -> List[Dict[str, Any]]:
        """
        Query Neo4j for PubMedArticle nodes that need embeddings.

        Returns:
            List of dictionaries containing article ID and abstract text
        """
        query = """
        MATCH (article:PubMedArticle)
        WHERE article.abstract IS NOT NULL 
          AND article.abstract_embedding IS NULL
        RETURN elementId(article) AS id, article.abstract AS abstract
        """

        with self.driver.session(database=self.neo4j_database) as session:
            result = session.run(query)
            articles = [dict(record) for record in result]

        return articles

    def process_batch(self, batch: List[Dict[str, Any]]) -> int:
        """
        Process a batch of articles: calculate embeddings and update Neo4j.

        Args:
            batch: List of article dictionaries with 'id' and 'abstract'

        Returns:
            Number of articles successfully processed
        """
        if not batch:
            return 0

        # Extract abstracts for batch embedding calculation
        abstracts = [article['abstract'] for article in batch]

        # Calculate embeddings for entire batch
        embeddings = self.embedding_model.encode(
            abstracts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Prepare data for Neo4j update
        timestamp = datetime.utcnow().isoformat()
        updates = []

        for i, article in enumerate(batch):
            updates.append({
                'id': article['id'],
                'embedding': embeddings[i].tolist(),
                'model': self.embedding_model_name,
                'timestamp': timestamp
            })

        # Update Neo4j in a single transaction
        query = """
        UNWIND $updates AS update
        MATCH (article:PubMedArticle)
        WHERE elementId(article) = update.id
        SET article.abstract_embedding = update.embedding,
            article.abstract_embedding_model = update.model,
            article.abstract_embedding_date = update.timestamp
        """

        with self.driver.session(database=self.neo4j_database) as session:
            result = session.run(query, updates=updates)
            result.consume()

        return len(batch)

    def process_all_articles(self, batch_size: int = 100):
        """
        Process all articles needing embeddings in batches.

        Args:
            batch_size: Number of articles to process per transaction (default: 100)
        """
        print("Fetching articles that need embeddings...")
        articles = self.get_articles_needing_embeddings()
        total_articles = len(articles)

        if total_articles == 0:
            print("No articles found that need embeddings.")
            return

        print(f"Found {total_articles} articles to process")
        print(f"Processing in batches of {batch_size}...")

        processed = 0

        # Process in batches
        for i in range(0, total_articles, batch_size):
            batch = articles[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_articles + batch_size - 1) // batch_size

            try:
                count = self.process_batch(batch)
                processed += count
                print(f"Batch {batch_num}/{total_batches}: Processed {count} articles "
                      f"(Total: {processed}/{total_articles})")
            except Exception as e:
                print(f"Error processing batch {batch_num}: {str(e)}", file=sys.stderr)
                continue

        print(f"\nProcessing complete: {processed}/{total_articles} articles updated")

    def close(self):
        """Close Neo4j driver connection."""
        self.driver.close()


def main():
    """Main execution function."""
    processor = None

    try:
        processor = PubMedEmbeddingProcessor()
        processor.process_all_articles(batch_size=100)

    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    finally:
        if processor:
            processor.close()
            print("Neo4j connection closed")


if __name__ == "__main__":
    main()