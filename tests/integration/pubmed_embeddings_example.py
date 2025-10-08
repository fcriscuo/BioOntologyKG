# An integration test to evaluate querying the PubMed embeddings in a Neo4j database
import logging
import os

from src.BioOntologyKG.embedding.PubMedAbstractEmbedding import PubMedEmbeddingsManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def main():

    # Configuration
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USER = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

    manager = PubMedEmbeddingsManager(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        embedding_model="all-mpnet-base-v2"  # 768 dimension embedding model
    )
    print("\nPerforming similarity search...")
    query = "synonymous mutations"
    results = manager.similarity_search(query, top_k=3)
    print(f"\nTop {len(results)} similar PubMedArticles for query: '{query}'")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. PMID: {result['pmid']} (Similarity: {result['similarity_score']:.4f})")
        print(f"   Title: {result['title']}")
        print(f"   First Author: {result['first_author']}")
        print(f"   Journal: {result['journal']} ({result['year']})")
        print(f"   Volume: {result['volume']}, Issue: {result['issue']}")
        print(f"   DOI: {result['doi']}")
        print(f"   Abstract: {result['abstract']}")

if __name__ == "__main__":
    main()