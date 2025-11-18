import os
import torch
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

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


def get_gene_count(driver):
    """Get total count of Gene nodes with ENTREZ_GENE_SUMMARY."""
    with driver.session() as session:
        result = session.run(
            "MATCH (g:Gene) WHERE g.ENTREZ_GENE_SUMMARY IS NOT NULL RETURN count(g) as count"
        )
        return result.single()['count']


def fetch_genes_batch(tx, skip, limit):
    """Fetch a batch of Gene nodes with their summaries."""
    query = """
    MATCH (g:Gene)
    WHERE g.ENTREZ_GENE_SUMMARY IS NOT NULL
    RETURN elementId(g) as id, g.ENTREZ_GENE_SUMMARY as summary
    SKIP $skip
    LIMIT $limit
    """
    result = tx.run(query, skip=skip, limit=limit)
    return [(record['id'], record['summary']) for record in result]


def update_gene_embedding(tx, gene_id, embedding):
    """Replace the gene_summary_embedding property with new embedding."""
    query = """
    MATCH (g:Gene)
    WHERE elementId(g) = $gene_id
    SET g.gene_summary_embedding = $embedding
    """
    tx.run(query, gene_id=gene_id, embedding=embedding)


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


def process_genes(driver, tokenizer, model, device):
    """Process all Gene nodes and replace embeddings with PubMedBERT embeddings."""
    total_genes = get_gene_count(driver)
    print(f"Found {total_genes} Gene nodes to process")

    processed = 0
    skip = 0

    with tqdm(total=total_genes, desc="Processing genes") as pbar:
        while skip < total_genes:
            # Fetch batch of genes
            with driver.session() as session:
                genes_batch = session.execute_read(
                    fetch_genes_batch, skip, BATCH_SIZE
                )

            if not genes_batch:
                break

            # Prepare summaries for embedding
            summaries = []
            gene_ids = []

            for gene_id, summary in genes_batch:
                # Truncate summary if needed
                truncated_summary = summary[:TEXT_CHAR_LIMIT] if summary else ""
                summaries.append(truncated_summary)
                gene_ids.append(gene_id)

            # Generate embeddings in batch
            embeddings = generate_embeddings(summaries, tokenizer, model, device)

            # Update nodes with new embeddings
            with driver.session() as session:
                for gene_id, embedding in zip(gene_ids, embeddings):
                    session.execute_write(
                        update_gene_embedding,
                        gene_id,
                        embedding.tolist()
                    )

            processed += len(genes_batch)
            skip += BATCH_SIZE
            pbar.update(len(genes_batch))

    print(f"\nSuccessfully processed {processed} Gene nodes")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Gene Summary PubMedBERT Embedding Replacement Process")
    print("=" * 60)
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
        # Process all genes
        print("Starting embedding replacement process...")
        print("This will REPLACE all existing gene_summary_embedding values\n")
        process_genes(driver, tokenizer, model, device)
        print("\nProcess completed successfully!")
        print("All gene_summary_embedding properties have been updated with PubMedBERT embeddings")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise

    finally:
        driver.close()
        print("Neo4j connection closed")


if __name__ == "__main__":
    main()