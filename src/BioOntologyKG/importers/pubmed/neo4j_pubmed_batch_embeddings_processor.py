#!/usr/bin/env python3
"""
Neo4j PubMed Batch Processor with PubMedBERT Embeddings
Queries Neo4j for PubMedArticle nodes with null titles, fetches their data from PubMed,
and generates abstract embeddings using PubMedBERT
"""

from Bio import Entrez
from neo4j import GraphDatabase
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Union
import time
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set your email for NCBI API (required by NCBI guidelines)
Entrez.email = os.getenv('NCBI_EMAIL') # Replace with your actual email


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
            'references': self._get_references(pubmed_data),
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


class PubMedBatchProcessor:
    """Process PubMed articles in batches from Neo4j with embeddings"""

    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 pubmed_email: str, batch_size: int = 10,
                 embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        """
        Initialize batch processor

        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            pubmed_email: Email for NCBI API
            batch_size: Number of records to process per batch
            embedding_model: PubMedBERT model name for embeddings
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.pubmed_fetcher = PubMedFetcher(pubmed_email)
        self.embedder = PubMedBERTEmbedder(model_name=embedding_model)
        self.batch_size = batch_size
        self.embedding_dimension = self.embedder.embedding_dim

        # Statistics
        self.total_processed = 0
        self.total_updated = 0
        self.total_failed = 0
        self.total_embeddings_generated = 0

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
            # remove leading blank
            pmids = [record['pubmed_id'].strip() for record in result]
            return pmids

    def update_pubmed_article(self, pubmed_data: Dict) -> bool:
        """
        Update PubMedArticle node with fetched data and embeddings

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

            # Generate embedding for abstract
            abstract_text = pubmed_data.get('abstract', '')
            abstract_embedding = None

            if abstract_text:
                try:
                    logger.info(f"Generating embedding for PMID: {pubmed_data.get('pmid')}")
                    abstract_embedding = self.embedder.encode(abstract_text).tolist()
                    self.total_embeddings_generated += 1
                    logger.info(f"Embedding generated successfully (dimension: {len(abstract_embedding)})")
                except Exception as e:
                    logger.error(f"Error generating embedding: {e}")
                    # Continue without embedding
            else:
                logger.warning(f"No abstract available for PMID: {pubmed_data.get('pmid')}")

            # Update node with all data including embedding
            update_query = """
            MERGE (p:PubMedArticle {pubmed_id: $pubmed_id})
            SET p.title = $title,
                p.abstract = $abstract,
                p.abstract_embedding = $abstract_embedding,
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
                p.references = $references,
                p.status = $status,
                p.dateCreated = $date_created,
                p.dateRevised = $date_revised,
                p.updatedAt = datetime()
            RETURN p.pubmed_id as pubmed_id
            """

            parameters = {
                'pubmed_id': str(pubmed_data.get('pmid', '')),
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
                'publication_types': pubmed_data.get('publication_types', []),
                'keywords': pubmed_data.get('keywords', []),
                'references': pubmed_data.get('references', []),
                'country': pubmed_data.get('country', ''),
                'status': pubmed_data.get('status', ''),
                'date_created': pubmed_data.get('date_created', ''),
                'date_revised': pubmed_data.get('date_revised', '')
            }

            with self.driver.session() as session:
                result = session.run(update_query, parameters)
                record = result.single()

                if record:
                    #if the original pubmed_id is a string that starts with a blank delete that node
                    #the PubMedArticle node that was updated has a pubmed_id property w/o a starting blank
                    original_pmid = ' ' +record['pubmed_id']
                    self.delete_invalid_pubmed_article(original_pmid)
                    logger.warning(f"*******Deleted PubMedArticle node with PMID:{original_pmid}")
                    logger.info(f"Updated PubMedArticle node for PMID: {record['pubmed_id']}")
                    return True
                else:
                    logger.error(f"Failed to update node for PMID: {pubmed_data.get('pmid')}")
                    return False

        except Exception as e:
            logger.error(f"Error updating PubMedArticle node: {e}")
            return False

    def delete_invalid_pubmed_article(self, pubmed_id: str) -> bool:
        """
        Delete a PubMedArticle node with an invalid pubmed_id

        Args:
            pubmed_id: The pubmed_id of the article to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        delete_query = """
        MATCH (p:PubMedArticle {pubmed_id: $pubmed_id})
        DETACH DELETE p
        RETURN count(p) as deleted_count
        """

        try:
            with self.driver.session() as session:
                result = session.run(delete_query, {'pubmed_id': str(pubmed_id)})
                record = result.single()

                if record and record['deleted_count'] > 0:
                    logger.info(f"Deleted invalid PubMedArticle node: {pubmed_id}")
                    return True
                else:
                    logger.warning(f"No PubMedArticle node found for pubmed_id: {pubmed_id}")
                    return False

        except Exception as e:
            logger.error(f"Error deleting PubMedArticle node {pubmed_id}: {e}")
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
        logger.info(f"+++++PMIDs: {pmids}")
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
                    self.delete_invalid_pubmed_article(pmid)
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
        logger.info(f"Embeddings generated: {self.total_embeddings_generated}")
        logger.info(f"Failed: {self.total_failed}")
        logger.info(f"{'=' * 60}")

    def create_vector_index(self, index_name: str = "pubmed_abstract_embeddings"):
        """
        Create vector index for abstract embeddings

        Args:
            index_name: Name for the vector index
        """
        create_index_query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (p:PubMedArticle)
        ON p.abstract_embedding
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
                logger.info(f"Index dimension: {self.embedding_dimension}")
                logger.info(f"Similarity function: cosine")
            except Exception as e:
                logger.error(f"Error creating vector index: {e}")

    def similarity_search(self, query_text: str, top_k: int = 5,
                          index_name: str = "pubmed_abstract_embeddings") -> List[Dict]:
        """
        Perform similarity search on PubMed abstracts

        Args:
            query_text: Search query text
            top_k: Number of results to return
            index_name: Name of the vector index

        Returns:
            List of similar publications with metadata and scores
        """
        try:
            # Generate embedding for query
            logger.info(f"Generating embedding for query: '{query_text[:50]}...'")
            query_embedding = self.embedder.encode(query_text).tolist()

            # Vector similarity search query
            search_query = f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
            YIELD node, score
            RETURN node.pubmed_id as pubmed_id,
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
                        'pubmed_id': record['pubmed_id'],
                        'title': record['title'],
                        'abstract': record['abstract'][:5000] + "..." if record['abstract'] and len(
                            record['abstract']) > 5000 else record['abstract'],
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


def main():
    """Main execution"""
    # Configuration
    # Configuration
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USER = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')  # Replace with your password
    PUBMED_EMAIL = os.getenv('NCBI_EMAIL')  # Replace with your email
    BATCH_SIZE = 10
    EMBEDDING_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

    # Initialize processor
    logger.info("Initializing PubMed batch processor with embeddings...")
    processor = PubMedBatchProcessor(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        pubmed_email=PUBMED_EMAIL,
        batch_size=BATCH_SIZE,
        embedding_model=EMBEDDING_MODEL
    )

    try:
        # Create vector index
        logger.info("\nCreating vector index for abstract embeddings...")
        processor.create_vector_index()

        # Process all batches
        logger.info("\nStarting batch processing...")
        processor.process_all()

        # Optional: Test similarity search
        logger.info("\n" + "=" * 60)
        logger.info("Testing similarity search...")
        logger.info("=" * 60)

        test_query = "cancer immunotherapy treatment and therapeutic targets"
        results = processor.similarity_search(test_query, top_k=3)

        if results:
            logger.info(f"\nTop {len(results)} similar articles for query: '{test_query}'")
            for i, result in enumerate(results, 1):
                logger.info(f"\n{i}. PMID: {result['pubmed_id']} (Similarity: {result['similarity_score']:.4f})")
                logger.info(f"   Title: {result['title']}")
                logger.info(f"   First Author: {result['first_author']}")
                logger.info(f"   Journal: {result['journal']} ({result['year']})")
                logger.info(f"   Volume: {result['volume']}, Issue: {result['issue']}")
        else:
            logger.info("No results found for similarity search")

    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        logger.info(f"Processed {processor.total_processed} records before interruption")
        logger.info(f"Generated {processor.total_embeddings_generated} embeddings")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

    finally:
        processor.close()
        logger.info("Neo4j connection closed")


if __name__ == "__main__":
    main()