#!/usr/bin/env python3
"""
PubMed Embeddings with Local Models and Neo4j Vector Search
Uses local embedding models instead of OpenAI to minimize costs
"""

import json
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PubMedEmbeddingsManager:
    """Manages PubMed data with local embeddings and Neo4j vector search"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize with Neo4j connection and embedding model

        Args:
            neo4j_uri: Neo4j database URI (e.g., "bolt://localhost:7687")
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            embedding_model: Local embedding model name
        """
        # Initialize Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # Initialize local embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()

        logger.info(f"Embedding dimension: {self.embedding_dimension}")

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

    def create_vector_index(self, index_name: str = "pubmed_abstract_embeddings"):
        """
        Create vector index for PubMed abstract embeddings

        Args:
            index_name: Name for the vector index
        """
        create_index_query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (p:PubMedArticle)
        ON p.abstractEmbedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {self.embedding_dimension},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """

        with self.driver.session() as session:
            try:
                session.run(create_index_query)
                logger.info(f"Vector index '{index_name}' created successfully")
            except Exception as e:
                logger.error(f"Error creating vector index: {e}")

    def create_PubMedArticle_node(self, pubmed_data: Dict) -> bool:
        """
        Create PubMedArticle node with embeddings in Neo4j

        Args:
            pubmed_data: PubMed data from the fetcher

        Returns:
            bool: Success status
        """
        try:
            # Generate embedding for abstract
            abstract_text = pubmed_data.get('abstract', '')
            if not abstract_text:
                logger.warning(f"No abstract found for PMID {pubmed_data.get('pmid')}")
                return False

            abstract_embedding = self.embedding_model.encode(abstract_text).tolist()

            # Prepare metadata
            first_author = ""
            if pubmed_data.get('authors') and len(pubmed_data['authors']) > 0:
                author = pubmed_data['authors'][0]
                first_author = f"{author.get('first_name', '')} {author.get('last_name', '')}".strip()

            journal_info = pubmed_data.get('journal', {})

            # Extract year from PubMedArticle date
            pub_date = pubmed_data.get('PubMedArticle_date', '')
            year = pub_date.split('-')[0] if pub_date else ''
            if not year and journal_info.get('pub_date'):
                year = journal_info['pub_date'].split()[0] if journal_info['pub_date'] else ''

            # Create node with embedding and metadata
            create_node_query = """
            MERGE (p:PubMedArticle {pmid: $pmid})
            SET p.title = $title,
                p.abstract = $abstract,
                p.abstractEmbedding = $abstract_embedding,
                p.firstAuthor = $first_author,
                p.journalTitle = $journal_title,
                p.volume = $volume,
                p.issue = $issue,
                p.yearPublished = $year,
                p.doi = $doi,
                p.language = $language,
                p.PubMedArticleTypes = $PubMedArticle_types,
                p.createdAt = datetime()
            RETURN p.pmid as pmid
            """

            parameters = {
                'pmid': str(pubmed_data.get('pmid', '')),
                'title': pubmed_data.get('title', ''),
                'abstract': abstract_text,
                'abstract_embedding': abstract_embedding,
                'first_author': first_author,
                'journal_title': journal_info.get('title', ''),
                'volume': journal_info.get('volume', ''),
                'issue': journal_info.get('issue', ''),
                'year': year,
                'doi': pubmed_data.get('doi', ''),
                'language': pubmed_data.get('language', []),
                'PubMedArticle_types': pubmed_data.get('PubMedArticle_types', [])
            }

            with self.driver.session() as session:
                result = session.run(create_node_query, parameters)
                record = result.single()

                if record:
                    logger.info(f"Created PubMedArticle node for PMID: {record['pmid']}")

                    # Create Author nodes and relationships
                    self._create_author_relationships(pubmed_data)

                    # Create MeSH term relationships
                    self._create_mesh_relationships(pubmed_data)

                    return True
                else:
                    logger.error(f"Failed to create node for PMID: {pubmed_data.get('pmid')}")
                    return False

        except Exception as e:
            logger.error(f"Error creating PubMedArticle node: {e}")
            return False

    def _create_author_relationships(self, pubmed_data: Dict):
        """Create Author nodes and relationships"""
        pmid = str(pubmed_data.get('pmid', ''))
        authors = pubmed_data.get('authors', [])

        if not authors:
            return

        create_authors_query = """
        MATCH (p:PubMedArticle {pmid: $pmid})
        UNWIND $authors as author_data
        MERGE (a:Author {
            firstName: author_data.first_name,
            lastName: author_data.last_name,
            initials: author_data.initials
        })
        ON CREATE SET a.orcid = author_data.orcid,
                     a.affiliation = author_data.affiliation
        MERGE (a)-[:AUTHORED {position: author_data.position}]->(p)
        """

        authors_data = []
        for i, author in enumerate(authors):
            authors_data.append({
                'first_name': author.get('first_name', ''),
                'last_name': author.get('last_name', ''),
                'initials': author.get('initials', ''),
                'orcid': author.get('orcid', ''),
                'affiliation': author.get('affiliation', ''),
                'position': i + 1
            })

        with self.driver.session() as session:
            session.run(create_authors_query, {'pmid': pmid, 'authors': authors_data})

    def _create_mesh_relationships(self, pubmed_data: Dict):
        """Create MeSH term nodes and relationships"""
        pmid = str(pubmed_data.get('pmid', ''))
        mesh_terms = pubmed_data.get('mesh_terms', [])

        if not mesh_terms:
            return

        create_mesh_query = """
        MATCH (p:PubMedArticle {pmid: $pmid})
        UNWIND $mesh_terms as mesh_data
        MERGE (m:MeshTerm {descriptor: mesh_data.descriptor})
        MERGE (p)-[:HAS_MESH_TERM {majorTopic: mesh_data.major_topic}]->(m)
        """

        mesh_data = []
        for mesh_term in mesh_terms:
            mesh_data.append({
                'descriptor': mesh_term.get('descriptor', ''),
                'major_topic': mesh_term.get('major_topic', False)
            })

        with self.driver.session() as session:
            session.run(create_mesh_query, {'pmid': pmid, 'mesh_terms': mesh_data})

    def similarity_search(self, query_text: str, top_k: int = 5,
                          index_name: str = "pubmed_abstract_embeddings") -> List[Dict]:
        """
        Perform similarity search on PubMed abstracts

        Args:
            query_text: Search query text
            top_k: Number of results to return
            index_name: Name of the vector index

        Returns:
            List of similar PubMedArticles with metadata and scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_model.encode(query_text).tolist()

            # Vector similarity search query
            search_query = f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
            YIELD node, score
            RETURN node.pmid as pmid,
                   node.title as title,
                   node.abstract as abstract,
                   node.firstAuthor as firstAuthor,
                   node.journalTitle as journalTitle,
                   node.volume as volume,
                   node.issue as issue,
                   node.yearPublished as yearPublished,
                   node.doi as doi,
                   score
            ORDER BY score DESC
            """

            with self.driver.session() as session:
                result = session.run(search_query, {
                    'index_name': index_name,
                    'top_k': top_k,
                    'query_embedding': query_embedding
                })

                results = []
                for record in result:
                    results.append({
                        'pmid': record['pmid'],
                        'title': record['title'],
                        'abstract': record['abstract'][:300] + "..." if len(record['abstract']) > 300 else record[
                            'abstract'],
                        'first_author': record['firstAuthor'],
                        'journal': record['journalTitle'],
                        'volume': record['volume'],
                        'issue': record['issue'],
                        'year': record['yearPublished'],
                        'doi': record['doi'],
                        'similarity_score': float(record['score'])
                    })

                return results

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []

    def get_PubMedArticle_stats(self) -> Dict:
        """Get statistics about PubMedArticles in the database"""
        stats_query = """
        MATCH (p:PubMedArticle)
        RETURN count(p) as total_PubMedArticles,
               count(p.abstractEmbedding) as PubMedArticles_with_embeddings,
               min(toInteger(p.yearPublished)) as earliest_year,
               max(toInteger(p.yearPublished)) as latest_year
        """

        with self.driver.session() as session:
            result = session.run(stats_query)
            record = result.single()

            if record:
                return {
                    'total_PubMedArticles': record['total_PubMedArticles'],
                    'PubMedArticles_with_embeddings': record['PubMedArticles_with_embeddings'],
                    'earliest_year': record['earliest_year'],
                    'latest_year': record['latest_year']
                }
            return {}


def main():
    """Example usage"""
    # Configuration
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USER = os.getenv('NEO4J_USER')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')  # Replace with your password

    # Initialize manager with local embedding model
    # Recommended local models:
    # - "all-MiniLM-L6-v2" (384 dimensions, fast and good quality)
    # - "all-mpnet-base-v2" (768 dimensions, higher quality)
    # - "dmis-lab/biobert-base-cased-v1.1" (768 dimensions, biomedical domain)

    manager = PubMedEmbeddingsManager(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        embedding_model="all-mpnet-base-v2"  # Change to biomedical model if desired
    )

    try:
        # Create vector index
        print("Creating vector index...")
        manager.create_vector_index()

        # Example: Load PubMed data (from your fetcher)
        # Replace with actual PubMed data
        example_pubmed_data = {
            "pmid": "33747033",
            "title": "Example biomedical research article",
            "abstract": "This study investigates the molecular mechanisms underlying cancer cell proliferation and identifies potential therapeutic targets for treatment.",
            "authors": [
                {"first_name": "John", "last_name": "Smith", "initials": "JS", "orcid": "",
                 "affiliation": "University Medical Center"},
                {"first_name": "Jane", "last_name": "Doe", "initials": "JD", "orcid": "",
                 "affiliation": "Research Institute"}
            ],
            "journal": {
                "title": "Nature Biotechnology",
                "volume": "39",
                "issue": "3"
            },
            "PubMedArticle_date": "2021-03-15",
            "doi": "10.1038/s41587-021-00123-4",
            "mesh_terms": [
                {"descriptor": "Neoplasms", "major_topic": True},
                {"descriptor": "Cell Proliferation", "major_topic": False}
            ]
        }

        # Create PubMedArticle node with embeddings
        print("Creating PubMedArticle node...")
        success = manager.create_PubMedArticle_node(example_pubmed_data)

        if success:
            print("PubMedArticle created successfully!")

            # Perform similarity search
            print("\nPerforming similarity search...")
            query = "cancer treatment and therapeutic targets"
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

        # Get database statistics
        print("\nDatabase Statistics:")
        stats = manager.get_PubMedArticle_stats()
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

    finally:
        manager.close()


if __name__ == "__main__":
    main()