import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


def get_embedding(text, model):
    """Generate embedding for the search text."""
    return model.encode(text).tolist()


def search_similar_abstracts(driver, search_embedding, top_k=5):
    """
    Search for similar abstracts using vector similarity.
    Returns top_k most similar results.
    """
    query = """
    CALL db.index.vector.queryNodes('pubmed_abstract_embeddings', $top_k, $search_embedding)
    YIELD node, score
    RETURN node.pubmed_id AS pmid,
           node.title AS title,
           node.abstract AS abstract,
           node.journal AS journal,
           node.publication_date AS pub_date,
           node.authors AS authors,
           score
    ORDER BY score DESC
    """

    with driver.session() as session:
        result = session.run(query, search_embedding=search_embedding, top_k=top_k)
        return [record.data() for record in result]


def display_results(results):
    """Display search results in a readable format."""
    if not results:
        print("\nNo results found.\n")
        return

    print(f"\n{'=' * 80}")
    print(f"Found {len(results)} similar articles:")
    print(f"{'=' * 80}\n")

    for idx, record in enumerate(results, 1):
        print(f"Result {idx} (Similarity Score: {record['score']:.4f})")
        print(f"{'-' * 80}")
        print(f"PMID: {record['pmid']}")
        print(f"Title: {record['title']}")
        print(f"Journal: {record.get('journal', 'N/A')}")
        print(f"Publication Date: {record.get('pub_date', 'N/A')}")

        if record.get('authors'):
            authors = record['authors']
            if isinstance(authors, list):
                authors_str = ', '.join(authors[:3])
                if len(authors) > 3:
                    authors_str += f" et al. ({len(authors)} total)"
            else:
                authors_str = authors
            print(f"Authors: {authors_str}")

        print(f"\nAbstract:")
        abstract = record.get('abstract', 'No abstract available')
        # Wrap abstract text for better readability
        print(f"{abstract[:2500]}{'...' if len(abstract) > 2500 else ''}")
        print(f"\n{'=' * 80}\n")


def main():
    # Get configuration from environment variables
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    embedding_model_name = os.getenv('PUBMED_EMBEDDING_MODEL',
                                     'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

    if not neo4j_password:
        print("Error: NEO4J_PASSWORD environment variable is required.")
        return

    print(f"Connecting to Neo4j at {neo4j_uri}...")
    print(f"Loading embedding model: {embedding_model_name}...")

    # Initialize Neo4j driverFF
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # Load embedding model
    try:
        model = SentenceTransformer(embedding_model_name)
        print(f"Model loaded successfully.\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        driver.close()
        return

    print("=" * 80)
    print("PubMed Abstract Similarity Search")
    print("=" * 80)
    print("Enter your search query to find similar abstracts.")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            # Prompt user for search string
            search_query = input("Search query: ").strip()

            # Check for exit command
            if search_query.lower() == 'exit':
                print("\nExiting. Goodbye!")
                break

            # Skip empty queries
            if not search_query:
                print("Please enter a search query.\n")
                continue

            try:
                # Generate embedding for search query
                print(f"\nSearching for: '{search_query}'...")
                search_embedding = get_embedding(search_query, model)

                # Perform similarity search
                results = search_similar_abstracts(driver, search_embedding, top_k=5)

                # Display results
                display_results(results)

            except Exception as e:
                print(f"\nError performing search: {e}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting...")

    finally:
        # Close Neo4j connection
        driver.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()