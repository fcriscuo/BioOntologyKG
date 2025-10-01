#!/usr/bin/env python3
"""
Neo4j PubMed Batch Processor
Queries Neo4j for PubMedArticle nodes with null titles and fetches their data from PubMed
"""

from Bio import Entrez
from neo4j import GraphDatabase
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional
import time
import logging
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set your email for NCBI API (required by NCBI guidelines)
Entrez.email = "your.email@example.com"  # Replace with your actual email


class PubMedFetcher:
    """Class to fetch and parse PubMed articles using BioPython"""

    def __init__(self, email: str):
        """Initialize with email for NCBI API"""
        Entrez.email = email

    def fetch_pubmed_article(self, pubmed_id: str) -> Optional[Dict]:
        """
        Fetch a full PubMed article by PMID and return structured data

        Args:
            pubmed_id (str): PubMed ID (PMID)

        Returns:
            dict: Structured article data or None if not found
        """
        try:
            # Fetch the article record
            handle = Entrez.efetch(db="pubmed", id=pubmed_id, retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            if not records['PubmedArticle']:
                logger.warning(f"No article found for PMID: {pubmed_id}")
                return None

            article = records['PubmedArticle'][0]
            return self._parse_article(article)

        except Exception as e:
            logger.error(f"Error fetching PMID {pubmed_id}: {str(e)}")
            return None

    def _parse_article(self, article: Dict) -> Dict:
        """Parse the raw article data into structured fields"""

        medline_citation = article['MedlineCitation']
        pubmed_data = article.get('PubmedData', {})

        # Basic article info
        article_data = medline_citation['Article']

        # Parse structured data
        parsed_data = {
            'pmid': str(medline_citation['PMID']),
            'title': self._get_title(article_data),
            'abstract': self._get_abstract(article_data),
            'authors': self._get_authors(article_data),
            'journal': self._get_journal_info(article_data),
            'publication_date': self._get_publication_date(article_data),
            'doi': self._get_doi(article_data),
            'keywords': self._get_keywords(medline_citation),
            'mesh_terms': self._get_mesh_terms(medline_citation),
            'publication_types': self._get_publication_types(article_data),
            'language': self._get_language(article_data),
            'country': self._get_country(medline_citation),
            'affiliations': self._get_affiliations(article_data),
            'grants': self._get_grants(article_data),
            'references_pmids': self._get_references(pubmed_data),
            'status': str(medline_citation.get('Status', '')),
            'date_created': self._get_date_created(medline_citation),
            'date_revised': self._get_date_revised(medline_citation)
        }

        return parsed_data

    def _get_title(self, article_data: Dict) -> str:
        """Extract article title"""
        return str(article_data.get('ArticleTitle', ''))

    def _get_abstract(self, article_data: Dict) -> str:
        """Extract and concatenate abstract text"""
        abstract = article_data.get('Abstract', {})
        if not abstract:
            return ''

        abstract_texts = abstract.get('AbstractText', [])
        if not abstract_texts:
            return ''

        # Handle single string
        if isinstance(abstract_texts, (str, type(abstract_texts))) and not isinstance(abstract_texts, list):
            return str(abstract_texts)

        # Handle list of abstract sections
        if isinstance(abstract_texts, list):
            full_abstract = []
            for text_section in abstract_texts:
                # Check if it's a StringElement with attributes (structured abstract)
                if hasattr(text_section, 'attributes') and text_section.attributes:
                    label = text_section.attributes.get('Label', '')
                    if label:
                        full_abstract.append(f"{label}: {str(text_section)}")
                    else:
                        full_abstract.append(str(text_section))
                else:
                    full_abstract.append(str(text_section))
            return ' '.join(full_abstract)

        return str(abstract_texts)

    def _get_authors(self, article_data: Dict) -> List[Dict]:
        """Extract author information"""
        authors = []
        author_list = article_data.get('AuthorList', [])

        for author in author_list:
            author_info = {
                'last_name': str(author.get('LastName', '')),
                'first_name': str(author.get('ForeName', '')),
                'initials': str(author.get('Initials', '')),
                'affiliation': author.get('AffiliationInfo', [{}])[0].get('Affiliation', '') if author.get(
                    'AffiliationInfo') else '',
                'orcid': self._extract_orcid(author)
            }
            authors.append(author_info)

        return authors

    def _extract_orcid(self, author: Dict) -> str:
        """Extract ORCID from author data"""
        identifiers = author.get('Identifier', [])
        for identifier in identifiers:
            if hasattr(identifier, 'attributes') and identifier.attributes:
                if identifier.attributes.get('Source') == 'ORCID':
                    return str(identifier)
        return ''

    def _get_journal_info(self, article_data: Dict) -> Dict:
        """Extract journal information"""
        journal = article_data.get('Journal', {})
        journal_issue = journal.get('JournalIssue', {})
        issn_info = journal.get('ISSN', {})

        return {
            'title': str(journal.get('Title', '')),
            'iso_abbreviation': str(journal.get('ISOAbbreviation', '')),
            'issn': str(issn_info) if issn_info else '',
            'volume': str(journal_issue.get('Volume', '')),
            'issue': str(journal_issue.get('Issue', '')),
            'pub_date': self._parse_journal_date(journal_issue)
        }

    def _parse_journal_date(self, journal_issue: Dict) -> str:
        """Parse journal publication date"""
        pub_date = journal_issue.get('PubDate', {})
        year = str(pub_date.get('Year', ''))
        month = str(pub_date.get('Month', ''))
        day = str(pub_date.get('Day', ''))

        date_parts = [part for part in [year, month, day] if part]
        return ' '.join(date_parts)

    def _get_publication_date(self, article_data: Dict) -> str:
        """Extract publication date"""
        article_date = article_data.get('ArticleDate', [])
        if article_date:
            date = article_date[0]
            year = str(date.get('Year', ''))
            month = str(date.get('Month', '')).zfill(2)
            day = str(date.get('Day', '')).zfill(2)
            return f"{year}-{month}-{day}" if all([year, month, day]) else year
        return ''

    def _get_doi(self, article_data: Dict) -> str:
        """Extract DOI"""
        elocation_id = article_data.get('ELocationID', [])
        for location in elocation_id:
            if hasattr(location, 'attributes'):
                if location.attributes.get('EIdType') == 'doi':
                    return str(location)
        return ''

    def _get_keywords(self, medline_citation: Dict) -> List[str]:
        """Extract keywords"""
        keyword_list = medline_citation.get('KeywordList', [])
        if not keyword_list:
            return []

        keywords = []
        for keyword_set in keyword_list:
            keywords.extend([str(kw) for kw in keyword_set])
        return keywords

    def _get_mesh_terms(self, medline_citation: Dict) -> List[Dict]:
        """Extract MeSH terms"""
        mesh_heading_list = medline_citation.get('MeshHeadingList', [])
        mesh_terms = []

        for mesh_heading in mesh_heading_list:
            descriptor = mesh_heading.get('DescriptorName', {})
            qualifiers = mesh_heading.get('QualifierName', [])

            mesh_term = {
                'descriptor': str(descriptor),
                'major_topic': False,
                'qualifiers': []
            }

            if hasattr(descriptor, 'attributes') and descriptor.attributes:
                mesh_term['major_topic'] = descriptor.attributes.get('MajorTopicYN') == 'Y'

            if qualifiers:
                for qual in qualifiers:
                    qualifier_info = {
                        'qualifier': str(qual),
                        'major_topic': False
                    }
                    if hasattr(qual, 'attributes') and qual.attributes:
                        qualifier_info['major_topic'] = qual.attributes.get('MajorTopicYN') == 'Y'
                    mesh_term['qualifiers'].append(qualifier_info)

            mesh_terms.append(mesh_term)

        return mesh_terms

    def _get_publication_types(self, article_data: Dict) -> List[str]:
        """Extract publication types"""
        pub_type_list = article_data.get('PublicationTypeList', [])
        return [str(pub_type) for pub_type in pub_type_list]

    def _get_language(self, article_data: Dict) -> List[str]:
        """Extract language information"""
        return [str(lang) for lang in article_data.get('Language', [])]

    def _get_country(self, medline_citation: Dict) -> str:
        """Extract country of publication"""
        medline_journal_info = medline_citation.get('MedlineJournalInfo', {})
        return str(medline_journal_info.get('Country', ''))

    def _get_affiliations(self, article_data: Dict) -> List[str]:
        """Extract all unique affiliations"""
        affiliations = set()
        author_list = article_data.get('AuthorList', [])

        for author in author_list:
            affiliation_info = author.get('AffiliationInfo', [])
            for affiliation in affiliation_info:
                if affiliation.get('Affiliation'):
                    affiliations.add(str(affiliation['Affiliation']))

        return list(affiliations)

    def _get_grants(self, article_data: Dict) -> List[Dict]:
        """Extract grant information"""
        grant_list = article_data.get('GrantList', [])
        grants = []

        for grant in grant_list:
            grant_info = {
                'grant_id': str(grant.get('GrantID', '')),
                'acronym': str(grant.get('Acronym', '')),
                'agency': str(grant.get('Agency', '')),
                'country': str(grant.get('Country', ''))
            }
            grants.append(grant_info)

        return grants

    def _get_references(self, pubmed_data: Dict) -> List[str]:
        """Extract referenced PMIDs"""
        reference_list = pubmed_data.get('ReferenceList', [])
        pmids = []

        for reference in reference_list:
            reference_info = reference.get('Reference', [])
            for ref in reference_info:
                article_id_list = ref.get('ArticleIdList', [])
                for article_id in article_id_list:
                    if hasattr(article_id, 'attributes') and article_id.attributes:
                        if article_id.attributes.get('IdType') == 'pubmed':
                            pmids.append(str(article_id))

        return pmids

    def _get_date_created(self, medline_citation: Dict) -> str:
        """Extract date created"""
        date_created = medline_citation.get('DateCreated', {})
        return self._format_date_dict(date_created)

    def _get_date_revised(self, medline_citation: Dict) -> str:
        """Extract date revised"""
        date_revised = medline_citation.get('DateRevised', {})
        return self._format_date_dict(date_revised)

    def _format_date_dict(self, date_dict: Dict) -> str:
        """Format a date dictionary to string"""
        if not date_dict:
            return ''

        year = str(date_dict.get('Year', ''))
        month = str(date_dict.get('Month', '')).zfill(2)
        day = str(date_dict.get('Day', '')).zfill(2)

        return f"{year}-{month}-{day}" if all([year, month, day]) else year


class PubMedBatchProcessor:
    """Process PubMed articles in batches from Neo4j"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 pubmed_email: str, batch_size: int = 10):
        """
        Initialize batch processor

        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            pubmed_email: Email for NCBI API
            batch_size: Number of records to process per batch
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.pubmed_fetcher = PubMedFetcher(pubmed_email)
        self.batch_size = batch_size

        # Statistics
        self.total_processed = 0
        self.total_updated = 0
        self.total_failed = 0

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

    def get_null_title_pmids(self) -> List[str]:
        """
        Query Neo4j for PubMedArticle nodes with null titles

        Returns:
            List of pubmed_id values (limited to batch_size)
        """
        query = """
        MATCH (p:PubMedArticle)
        WHERE p.title IS NULL
        RETURN p.pubmed_id as pubmed_id
        LIMIT $batch_size
        """

        with self.driver.session() as session:
            result = session.run(query, {'batch_size': self.batch_size})
            pmids = [record['pubmed_id'] for record in result]
            return pmids

    def update_pubmed_article(self, pubmed_data: Dict) -> bool:
        """
        Update PubMedArticle node with fetched data

        Args:
            pubmed_data: Fetched PubMed article data

        Returns:
            bool: Success status
        """
        try:
            # Prepare first author
            first_author = ""
            if pubmed_data.get('authors') and len(pubmed_data['authors']) > 0:
                author = pubmed_data['authors'][0]
                first_author = f"{author.get('first_name', '')} {author.get('last_name', '')}".strip()

            journal_info = pubmed_data.get('journal', {})

            # Extract year from publication date
            pub_date = pubmed_data.get('publication_date', '')
            year = pub_date.split('-')[0] if pub_date else ''
            if not year and journal_info.get('pub_date'):
                year = journal_info['pub_date'].split()[0] if journal_info['pub_date'] else ''

            # Update node
            update_query = """
            MATCH (p:PubMedArticle {pubmed_id: $pubmed_id})
            SET p.title = $title,
                p.abstract = $abstract,
                p.firstAuthor = $first_author,
                p.journalTitle = $journal_title,
                p.volume = $volume,
                p.issue = $issue,
                p.yearPublished = $year,
                p.doi = $doi,
                p.language = $language,
                p.publicationTypes = $publication_types,
                p.keywords = $keywords,
                p.country = $country,
                p.status = $status,
                p.dateCreated = $date_created,
                p.dateRevised = $date_revised,
                p.updatedAt = datetime()
            RETURN p.pubmed_id as pubmed_id
            """

            parameters = {
                'pubmed_id': str(pubmed_data.get('pmid', '')),
                'title': pubmed_data.get('title', ''),
                'abstract': pubmed_data.get('abstract', ''),
                'first_author': first_author,
                'journal_title': journal_info.get('title', ''),
                'volume': journal_info.get('volume', ''),
                'issue': journal_info.get('issue', ''),
                'year': year,
                'doi': pubmed_data.get('doi', ''),
                'language': pubmed_data.get('language', []),
                'publication_types': pubmed_data.get('publication_types', []),
                'keywords': pubmed_data.get('keywords', []),
                'country': pubmed_data.get('country', ''),
                'status': pubmed_data.get('status', ''),
                'date_created': pubmed_data.get('date_created', ''),
                'date_revised': pubmed_data.get('date_revised', '')
            }

            with self.driver.session() as session:
                result = session.run(update_query, parameters)
                record = result.single()

                if record:
                    logger.info(f"Updated PubMedArticle node for PMID: {record['pubmed_id']}")
                    return True
                else:
                    logger.error(f"Failed to update node for PMID: {pubmed_data.get('pmid')}")
                    return False

        except Exception as e:
            logger.error(f"Error updating PubMedArticle node: {e}")
            return False

    def process_batch(self) -> int:
        """
        Process one batch of PubMedArticle nodes

        Returns:
            Number of records processed in this batch
        """
        # Get PMIDs with null titles
        pmids = self.get_null_title_pmids()

        if not pmids:
            logger.info("No more PubMedArticle nodes with null titles found")
            return 0

        logger.info(f"Processing batch of {len(pmids)} PMIDs")

        for pmid in pmids:
            try:
                logger.info(f"Fetching data for PMID: {pmid}")

                # Fetch PubMed data
                pubmed_data = self.pubmed_fetcher.fetch_pubmed_article(pmid)

                if pubmed_data:
                    # Update Neo4j node
                    success = self.update_pubmed_article(pubmed_data)

                    if success:
                        self.total_updated += 1
                    else:
                        self.total_failed += 1
                else:
                    logger.warning(f"No data retrieved for PMID: {pmid}")
                    self.total_failed += 1

                self.total_processed += 1

                # Rate limiting: NCBI allows 3 requests per second without API key
                # Add a small delay to be respectful
                time.sleep(0.34)  # ~3 requests per second

            except Exception as e:
                logger.error(f"Error processing PMID {pmid}: {e}")
                self.total_failed += 1
                self.total_processed += 1

        return len(pmids)

    def process_all(self):
        """Process all PubMedArticle nodes with null titles"""
        logger.info("Starting batch processing of PubMedArticle nodes")
        logger.info(f"Batch size: {self.batch_size}")

        batch_count = 0

        while True:
            batch_count += 1
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing batch #{batch_count}")
            logger.info(f"{'=' * 60}")

            records_processed = self.process_batch()

            if records_processed == 0:
                break

            logger.info(f"Batch #{batch_count} complete. Processed {records_processed} records.")
            logger.info(
                f"Progress - Total: {self.total_processed}, Updated: {self.total_updated}, Failed: {self.total_failed}")

        # Final summary
        logger.info(f"\n{'=' * 60}")
        logger.info("Batch processing complete!")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total batches processed: {batch_count - 1}")
        logger.info(f"Total records processed: {self.total_processed}")
        logger.info(f"Successfully updated: {self.total_updated}")
        logger.info(f"Failed: {self.total_failed}")
        logger.info(f"{'=' * 60}")


def main():
    """Main execution"""
    # Configuration
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USER = os.getenv('NEO4J_USER')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')  # Replace with your password
    PUBMED_EMAIL = os.getenv('NCBI_EMAIL') # Replace with your email
    BATCH_SIZE = 10

    # Initialize processor
    processor = PubMedBatchProcessor(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        pubmed_email=PUBMED_EMAIL,
        batch_size=BATCH_SIZE
    )

    try:
        # Process all batches
        processor.process_all()

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        logger.info(f"Processed {processor.total_processed} records before interruption")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    finally:
        processor.close()
        logger.info("Neo4j connection closed")


if __name__ == "__main__":
    main()