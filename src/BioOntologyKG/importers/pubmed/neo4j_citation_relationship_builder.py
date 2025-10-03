#!/usr/bin/env python3
"""
Neo4j Citation Relationship Builder
Creates CITES relationships between PubMedArticle nodes based on their references
"""

from neo4j import GraphDatabase
from typing import Dict, List, Optional, Set
import logging
import time
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CitationRelationshipBuilder:
    """Build citation relationships between PubMed articles"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 batch_size: int = 100):
        """
        Initialize citation builder

        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            batch_size: Number of articles to process per batch
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.batch_size = batch_size

        # Statistics
        self.total_articles_processed = 0
        self.total_citations_created = 0
        self.total_new_articles_created = 0
        self.total_articles_with_references = 0
        self.start_time = None

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

    def get_article_count(self) -> int:
        """Get total number of PubMedArticle nodes"""
        query = """
        MATCH (p:PubMedArticle)
        RETURN count(p) as total
        """

        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            return record['total'] if record else 0

    def get_articles_with_references_count(self) -> int:
        """Get count of articles that have references"""
        query = """
        MATCH (p:PubMedArticle)
        WHERE p.references_pmids IS NOT NULL 
          AND size(p.references_pmids) > 0
        RETURN count(p) as total
        """

        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            return record['total'] if record else 0

    def get_articles_batch(self, skip: int) -> List[Dict]:
        """
        Get a batch of articles with their references

        Args:
            skip: Number of records to skip

        Returns:
            List of article data with pubmed_id and references
        """
        query = """
        MATCH (p:PubMedArticle)
        WHERE p.references_pmids IS NOT NULL 
          AND size(p.references_pmids) > 0
        RETURN p.pubmed_id as pubmed_id, 
               p.references_pmids as references
        ORDER BY p.pubmed_id
        SKIP $skip
        LIMIT $batch_size
        """

        with self.driver.session() as session:
            result = session.run(query, {'skip': skip, 'batch_size': self.batch_size})
            articles = []
            for record in result:
                articles.append({
                    'pubmed_id': record['pubmed_id'],
                    'references': record['references'] if record['references'] else []
                })
            return articles

    def create_citation_relationships_batch(self, articles: List[Dict]) -> Dict[str, int]:
        """
        Create citation relationships for a batch of articles

        Args:
            articles: List of article dictionaries with pubmed_id and references

        Returns:
            Dictionary with statistics (citations_created, articles_created)
        """
        # Prepare batch data
        batch_data = []
        for article in articles:
            if article['references']:
                batch_data.append({
                    'citing_pmid': article['pubmed_id'],
                    'cited_pmids': article['references']
                })

        if not batch_data:
            return {'citations_created': 0, 'articles_created': 0}

        # Create relationships in a single transaction
        query = """
        UNWIND $batch_data as article_data
        MATCH (citing:PubMedArticle {pubmed_id: article_data.citing_pmid})
        UNWIND article_data.cited_pmids as cited_pmid

        // Create cited article if it doesn't exist
        MERGE (cited:PubMedArticle {pubmed_id: cited_pmid})
        ON CREATE SET cited.createdAt = datetime(),
                      cited.createdBy = 'citation_builder'

        // Create CITES relationship (avoid duplicates)
        MERGE (citing)-[r:CITES]->(cited)
        ON CREATE SET r.createdAt = datetime()

        RETURN count(DISTINCT cited_pmid) as citations_created,
               sum(CASE WHEN cited.createdBy = 'citation_builder' THEN 1 ELSE 0 END) as new_articles
        """

        try:
            with self.driver.session() as session:
                result = session.run(query, {'batch_data': batch_data})
                record = result.single()

                if record:
                    return {
                        'citations_created': record['citations_created'],
                        'articles_created': record['new_articles']
                    }
                else:
                    return {'citations_created': 0, 'articles_created': 0}

        except Exception as e:
            logger.error(f"Error creating citation relationships: {e}")
            return {'citations_created': 0, 'articles_created': 0}

    def process_all_citations(self):
        """Process all articles and create citation relationships"""
        self.start_time = datetime.now()

        # Get total counts
        total_articles = self.get_article_count()
        articles_with_refs = self.get_articles_with_references_count()

        logger.info("=" * 70)
        logger.info("Citation Relationship Builder Started")
        logger.info("=" * 70)
        logger.info(f"Total PubMedArticle nodes: {total_articles:,}")
        logger.info(f"Articles with references: {articles_with_refs:,}")
        logger.info(f"Articles without references: {total_articles - articles_with_refs:,}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info("=" * 70)

        if articles_with_refs == 0:
            logger.warning("No articles with references found. Nothing to process.")
            return

        # Process in batches
        skip = 0
        batch_num = 0

        while True:
            batch_num += 1

            # Get batch of articles
            articles = self.get_articles_batch(skip)

            if not articles:
                logger.info("No more articles to process")
                break

            logger.info(f"\n{'=' * 70}")
            logger.info(f"Processing Batch #{batch_num} (articles {skip + 1:,} to {skip + len(articles):,})")
            logger.info(f"{'=' * 70}")

            # Count total references in this batch
            total_refs_in_batch = sum(len(a['references']) for a in articles)
            logger.info(f"Articles in batch: {len(articles)}")
            logger.info(f"Total references in batch: {total_refs_in_batch:,}")

            # Process batch
            batch_start = time.time()
            stats = self.create_citation_relationships_batch(articles)
            batch_time = time.time() - batch_start

            # Update statistics
            self.total_articles_processed += len(articles)
            self.total_citations_created += stats['citations_created']
            self.total_new_articles_created += stats['articles_created']
            self.total_articles_with_references += len(articles)

            # Log batch results
            logger.info(f"Batch completed in {batch_time:.2f} seconds")
            logger.info(f"Citations created: {stats['citations_created']:,}")
            logger.info(f"New article nodes created: {stats['articles_created']:,}")

            # Calculate progress
            progress_pct = (self.total_articles_processed / articles_with_refs) * 100
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            articles_per_sec = self.total_articles_processed / elapsed_time if elapsed_time > 0 else 0

            # Estimate time remaining
            if articles_per_sec > 0:
                remaining_articles = articles_with_refs - self.total_articles_processed
                est_remaining_sec = remaining_articles / articles_per_sec
                est_remaining_min = est_remaining_sec / 60

                logger.info(
                    f"\nProgress: {progress_pct:.2f}% ({self.total_articles_processed:,}/{articles_with_refs:,})")
                logger.info(f"Processing rate: {articles_per_sec:.2f} articles/sec")
                logger.info(f"Estimated time remaining: {est_remaining_min:.1f} minutes")

            # Cumulative statistics
            logger.info(f"\nCumulative Statistics:")
            logger.info(f"  Total articles processed: {self.total_articles_processed:,}")
            logger.info(f"  Total citations created: {self.total_citations_created:,}")
            logger.info(f"  Total new articles created: {self.total_new_articles_created:,}")

            # Move to next batch
            skip += len(articles)

            # Small delay to prevent overwhelming the database
            time.sleep(0.1)

        # Final summary
        total_time = (datetime.now() - self.start_time).total_seconds()
        self._print_final_summary(total_time)

    def _print_final_summary(self, total_time: float):
        """Print final processing summary"""
        logger.info("\n" + "=" * 70)
        logger.info("Citation Relationship Building Complete!")
        logger.info("=" * 70)
        logger.info(f"Total processing time: {total_time / 60:.2f} minutes ({total_time:.1f} seconds)")
        logger.info(f"Total articles processed: {self.total_articles_processed:,}")
        logger.info(f"Total citation relationships created: {self.total_citations_created:,}")
        logger.info(f"Total new article nodes created: {self.total_new_articles_created:,}")

        if self.total_articles_processed > 0:
            avg_citations = self.total_citations_created / self.total_articles_processed
            logger.info(f"Average citations per article: {avg_citations:.2f}")

        if total_time > 0:
            articles_per_sec = self.total_articles_processed / total_time
            citations_per_sec = self.total_citations_created / total_time
            logger.info(f"Processing rate: {articles_per_sec:.2f} articles/sec")
            logger.info(f"Citation creation rate: {citations_per_sec:.2f} citations/sec")

        logger.info("=" * 70)

    def verify_citations(self) -> Dict[str, int]:
        """
        Verify citation relationships and gather statistics

        Returns:
            Dictionary with verification statistics
        """
        logger.info("\n" + "=" * 70)
        logger.info("Verifying Citation Relationships")
        logger.info("=" * 70)

        queries = {
            'total_articles': """
                MATCH (p:PubMedArticle)
                RETURN count(p) as count
            """,
            'articles_with_outgoing_citations': """
                MATCH (p:PubMedArticle)-[:CITES]->()
                RETURN count(DISTINCT p) as count
            """,
            'articles_with_incoming_citations': """
                MATCH ()-[:CITES]->(p:PubMedArticle)
                RETURN count(DISTINCT p) as count
            """,
            'total_citation_relationships': """
                MATCH ()-[r:CITES]->()
                RETURN count(r) as count
            """,
            'articles_created_by_builder': """
                MATCH (p:PubMedArticle)
                WHERE p.createdBy = 'citation_builder'
                RETURN count(p) as count
            """,
            'max_citations_per_article': """
                MATCH (p:PubMedArticle)-[r:CITES]->()
                WITH p, count(r) as citation_count
                RETURN max(citation_count) as count
            """,
            'max_cited_count': """
                MATCH ()-[r:CITES]->(p:PubMedArticle)
                WITH p, count(r) as cited_count
                RETURN max(cited_count) as count
            """
        }

        stats = {}

        with self.driver.session() as session:
            for key, query in queries.items():
                try:
                    result = session.run(query)
                    record = result.single()
                    stats[key] = record['count'] if record and record['count'] else 0
                except Exception as e:
                    logger.error(f"Error running query for {key}: {e}")
                    stats[key] = 0

        # Print verification results
        logger.info(f"Total PubMedArticle nodes: {stats['total_articles']:,}")
        logger.info(f"Articles with outgoing citations: {stats['articles_with_outgoing_citations']:,}")
        logger.info(
            f"Articles with incoming citations (cited by others): {stats['articles_with_incoming_citations']:,}")
        logger.info(f"Total CITES relationships: {stats['total_citation_relationships']:,}")
        logger.info(f"Articles created by citation builder: {stats['articles_created_by_builder']:,}")
        logger.info(f"Maximum citations from a single article: {stats['max_citations_per_article']:,}")
        logger.info(f"Maximum times an article is cited: {stats['max_cited_count']:,}")
        logger.info("=" * 70)

        return stats

    def find_most_cited_articles(self, top_n: int = 10) -> List[Dict]:
        """
        Find the most cited articles

        Args:
            top_n: Number of top articles to return

        Returns:
            List of articles with citation counts
        """
        query = """
        MATCH (p:PubMedArticle)<-[r:CITES]-()
        WITH p, count(r) as citation_count
        ORDER BY citation_count DESC
        LIMIT $top_n
        RETURN p.pubmed_id as pubmed_id,
               p.title as title,
               p.yearPublished as year,
               p.firstAuthor as first_author,
               p.journalTitle as journal,
               citation_count
        """

        with self.driver.session() as session:
            result = session.run(query, {'top_n': top_n})
            articles = []
            for record in result:
                articles.append({
                    'pubmed_id': record['pubmed_id'],
                    'title': record['title'] if record['title'] else 'N/A',
                    'year': record['year'] if record['year'] else 'N/A',
                    'first_author': record['first_author'] if record['first_author'] else 'N/A',
                    'journal': record['journal'] if record['journal'] else 'N/A',
                    'citation_count': record['citation_count']
                })
            return articles

    def find_articles_with_most_references(self, top_n: int = 10) -> List[Dict]:
        """
        Find articles that cite the most other articles

        Args:
            top_n: Number of top articles to return

        Returns:
            List of articles with reference counts
        """
        query = """
        MATCH (p:PubMedArticle)-[r:CITES]->()
        WITH p, count(r) as reference_count
        ORDER BY reference_count DESC
        LIMIT $top_n
        RETURN p.pubmed_id as pubmed_id,
               p.title as title,
               p.yearPublished as year,
               p.firstAuthor as first_author,
               p.journalTitle as journal,
               reference_count
        """

        with self.driver.session() as session:
            result = session.run(query, {'top_n': top_n})
            articles = []
            for record in result:
                articles.append({
                    'pubmed_id': record['pubmed_id'],
                    'title': record['title'] if record['title'] else 'N/A',
                    'year': record['year'] if record['year'] else 'N/A',
                    'first_author': record['first_author'] if record['first_author'] else 'N/A',
                    'journal': record['journal'] if record['journal'] else 'N/A',
                    'reference_count': record['reference_count']
                })
            return articles


def main():
    """Main execution"""
    # Configuration
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USER = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')  # Replace with your password
    PUBMED_EMAIL = os.getenv('NCBI_EMAIL')

    BATCH_SIZE = 100  # Process 100 articles at a time

    logger.info("Initializing Citation Relationship Builder...")

    builder = CitationRelationshipBuilder(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        batch_size=BATCH_SIZE
    )

    try:
        # Process all citations
        builder.process_all_citations()

        # Verify results
        logger.info("\nRunning verification checks...")
        stats = builder.verify_citations()

        # Find most cited articles
        logger.info("\n" + "=" * 70)
        logger.info("Top 10 Most Cited Articles")
        logger.info("=" * 70)
        most_cited = builder.find_most_cited_articles(10)
        for i, article in enumerate(most_cited, 1):
            logger.info(f"\n{i}. PMID: {article['pubmed_id']} ({article['citation_count']:,} citations)")
            logger.info(f"   Title: {article['title'][:80]}...")
            logger.info(f"   Author: {article['first_author']}")
            logger.info(f"   Journal: {article['journal']} ({article['year']})")

        # Find articles with most references
        logger.info("\n" + "=" * 70)
        logger.info("Top 10 Articles with Most References")
        logger.info("=" * 70)
        most_refs = builder.find_articles_with_most_references(10)
        for i, article in enumerate(most_refs, 1):
            logger.info(f"\n{i}. PMID: {article['pubmed_id']} ({article['reference_count']:,} references)")
            logger.info(f"   Title: {article['title'][:80]}...")
            logger.info(f"   Author: {article['first_author']}")
            logger.info(f"   Journal: {article['journal']} ({article['year']})")

    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user")
        logger.info(f"Articles processed: {builder.total_articles_processed:,}")
        logger.info(f"Citations created: {builder.total_citations_created:,}")
        logger.info(f"New articles created: {builder.total_new_articles_created:,}")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    finally:
        builder.close()
        logger.info("\nNeo4j connection closed")


if __name__ == "__main__":
    main()