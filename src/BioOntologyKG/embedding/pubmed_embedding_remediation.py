import os
import torch
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from datetime import datetime

# Configuration
BATCH_SIZE = 32  # Smaller batch for transformer model
TEXT_CHAR_LIMIT = 2000
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings to get sentence embedding.
    Takes attention mask into account for correct averaging.
    """
    token_embeddings = model_output[0]  # First element contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def connect_to_neo4j():
    """Establish connection to Neo4j database using environment variables."""
    uri = os.getenv('NEO4J_URI')
    user = os.getenv('NEO4J_USERNAME')
    password = os.getenv('NEO4J_PASSWORD')

    if not all([uri, user, password]):
        raise ValueError("Missing required environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver


def get_article_count(driver):
    """Get total count of PubMedArticle nodes with abstracts."""
    with driver.session() as session:
        result = session.run(
            "MATCH (p:PubMedArticle) WHERE p.abstract IS NOT NULL RETURN count(p) as count"
        )
        return result.single()['count']


def fetch_articles_batch(tx, skip, limit):
    """Fetch a batch of PubMedArticle nodes with their abstracts."""
    query = """
    MATCH (p:PubMedArticle)
    WHERE p.abstract IS NOT NULL
    RETURN elementId(p) as id, p.abstract as abstract_text
    SKIP $skip
    LIMIT $limit
    """
    result = tx.run(query, skip=skip, limit=limit)
    return [(record['id'], record['abstract_text']) for record in result]


def update_article_embedding(tx, article_id, embedding, model_name, timestamp):
    """Replace the abstract_embedding property and add metadata."""
    query = """
    MATCH (p:PubMedArticle)
    WHERE elementId(p) = $article_id
    SET p.abstract_embedding = $embedding,
        p.abstract_embedding_model = $model_name,
        p.abstract_embedding_date = $timestamp
    """
    tx.run(query, article_id=article_id, embedding=embedding,
           model_name=model_name, timestamp=timestamp)


def generate_embeddings(texts, tokenizer, model, device):
    """Generate embeddings using PubMedBERT."""
    # Tokenize sentences
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,  # BERT max sequence length
        return_tensors='pt'
    )

    # Move to device
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}

    # Generate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform mean pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings (optional but recommended for similarity search)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


def process_articles(driver, tokenizer, model, device):
    """Process all PubMedArticle nodes and replace embeddings with PubMedBERT embeddings."""
    total_articles = get_article_count(driver)
    print(f"Found {total_articles} PubMedArticle nodes to process")

    if total_articles == 0:
        print("No PubMedArticle nodes with abstracts found. Exiting.")
        return

    processed = 0
    skip = 0
    timestamp = datetime.utcnow().isoformat()

    with tqdm(total=total_articles, desc="Processing articles") as pbar:
        while skip < total_articles:
            # Fetch batch of articles
            with driver.session() as session:
                articles_batch = session.execute_read(
                    fetch_articles_batch, skip, BATCH_SIZE
                )

            if not articles_batch:
                break

            # Prepare abstracts for embedding
            abstracts = []
            article_ids = []

            for article_id, abstract_text in articles_batch:
                # Truncate abstract if needed
                truncated_abstract = abstract_text[:TEXT_CHAR_LIMIT] if abstract_text else ""
                abstracts.append(truncated_abstract)
                article_ids.append(article_id)

            # Generate embeddings in batch
            embeddings = generate_embeddings(abstracts, tokenizer, model, device)

            # Update nodes with new embeddings and metadata
            with driver.session() as session:
                for article_id, embedding in zip(article_ids, embeddings):
                    session.execute_write(
                        update_article_embedding,
                        article_id,
                        embedding.tolist(),
                        MODEL_NAME,
                        timestamp
                    )

            processed += len(articles_batch)
            skip += BATCH_SIZE
            pbar.update(len(articles_batch))

    print(f"\nSuccessfully processed {processed} PubMedArticle nodes")


def display_summary(driver):
    """Display summary of embedding status after processing."""
    with driver.session() as session:
        # Count articles with new embeddings
        result = session.run("""
            MATCH (p:PubMedArticle)
            WHERE p.abstract_embedding_model = $model_name
            RETURN count(p) as count
        """, model_name=MODEL_NAME)

        count = result.single()['count']
        print(f"\nPubMedArticle nodes with PubMedBERT embeddings: {count}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("PubMedArticle Abstract PubMedBERT Embedding Regeneration Process")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Text character limit: {TEXT_CHAR_LIMIT}")
    print(f"Batch size: {BATCH_SIZE}")

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load tokenizer and model
    print(f"Loading PubMedBERT model and tokenizer...")
    print("(This may take a few minutes on first run)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully\n")

    # Connect to Neo4j
    print("Connecting to Neo4j...")
    driver = connect_to_neo4j()
    print("Connected successfully\n")

    try:
        # Process all articles
        print("Starting embedding regeneration process...")
        print("This will REPLACE ALL existing abstract_embedding values")
        print("New properties will be added:")
        print("  - abstract_embedding_model: Model identifier")
        print("  - abstract_embedding_date: Generation timestamp\n")

        process_articles(driver, tokenizer, model, device)

        print("\nProcess completed successfully!")
        print("All abstract_embedding properties have been updated with PubMedBERT embeddings")

        # Display summary
        display_summary(driver)

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise

    finally:
        driver.close()
        print("\nNeo4j connection closed")


if __name__ == "__main__":
    main()