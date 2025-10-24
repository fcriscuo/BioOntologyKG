"""
Neo4j Cancer Knowledge Graph Builder
Mines PubMed citations and populates Neo4j with nodes, embeddings, and CITES relationships
"""

import json
import os
import sys
import time
from typing import Set, Dict, List, Optional
from xml.etree import ElementTree as ET

import requests
import torch
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel


# Initialize manager with local embedding model
# Recommended local models:
# - "all-MiniLM-L6-v2" (384 dimensions, fast and good quality)
# - "all-mpnet-base-v2" (768 dimensions, higher quality)
# - "dmis-lab/biobert-base-cased-v1.1" (768 dimensions, biomedical domain)

class Neo4jCancerKGBuilder:
    def __init__(
            self,
            neo4j_uri: str,
            neo4j_user: str,
            neo4j_password: str,
            pubmed_email: str,
            embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            #embedding_model: str = "all-mpnet-base-v2",
            tool_name: str = "CancerKnowledgeGraphBuilder"
    ):
        """
        Initialize the Cancer Knowledge Graph Builder.

        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            pubmed_email: Email for NCBI E-utilities
            embedding_model: HuggingFace model for embeddings
            tool_name: Tool name for API requests
        """
        # Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        # API configuration
        self.pubmed_email = pubmed_email
        self.tool_name = tool_name
        self.ncbi_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.europepmc_base = "https://www.ebi.ac.uk/europepmc/webservices/rest"
        self.crossref_base = "https://api.crossref.org/works"

        # Rate limiting
        self.ncbi_delay = 0.34
        self.europepmc_delay = 0.2
        self.crossref_delay = 1.0

        # Caching
        self.doi_to_pmid_cache = {}
        self.pmid_to_doi_cache = {}
        self.processed_pmids = set()  # Track what's already in Neo4j

        # Failed fetch logging
        self.failed_fetches = []  # List of (pmid, classification, reason)

        # Embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.model.eval()

        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Initialize Neo4j schema
        self._setup_neo4j_schema()

    def _setup_neo4j_schema(self):
        """Create indexes and constraints in Neo4j."""
        with self.driver.session() as session:
            # Create constraint on pubmed_id (ensures uniqueness)
            session.run("""
                        CREATE CONSTRAINT pubmed_id_unique IF NOT EXISTS
                        FOR (p:PubMedArticle) REQUIRE p.pubmed_id IS UNIQUE
                        """)

            # Create index on title for faster lookups
            session.run("""
                CREATE INDEX pubmed_title_idx IF NOT EXISTS
                FOR (p:PubMedArticle) ON (p.title)
            """)

            print("Neo4j schema initialized successfully")
            print("Note: Using existing vector index on 'abstract_embedding' property")

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()

    def check_node_exists(self, pmid: str) -> bool:
        """
        Check if a PubMedArticle node exists and has a title.

        Args:
            pmid: PubMed ID

        Returns:
            True if node exists with non-null title
        """
        with self.driver.session() as session:
            result = session.run("""
                                 MATCH (p:PubMedArticle {pubmed_id: $pmid})
                                 RETURN p.title IS NOT NULL AS has_title
                                 """, pmid=pmid)
            record = result.single()
            return record and record['has_title']

    def create_empty_node(self, pmid: str):
        """Create an empty PubMedArticle node as a placeholder."""
        with self.driver.session() as session:
            session.run("""
                        MERGE (p:PubMedArticle {pubmed_id: $pmid})
                        """, pmid=pmid)

    def delete_empty_node(self, pmid: str):
        """Delete a PubMedArticle node that has no title (failed fetch)."""
        with self.driver.session() as session:
            session.run("""
                        MATCH (p:PubMedArticle {pubmed_id: $pmid})
                          WHERE p.title IS NULL
                        DETACH DELETE p
                        """, pmid=pmid)
            print(f"  Deleted empty node for PMID:{pmid}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using BiomedBERT.

        Args:
            text: Input text (abstract)

        Returns:
            768-dimensional embedding vector
        """
        if not text or text.strip() == "":
            return [0.0] * 768  # Return zero vector for empty text

        try:
            # Tokenize and truncate to model's max length
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

            return embedding.tolist()
        except Exception as e:
            print(f"  Error generating embedding: {e}")
            return [0.0] * 768

    def fetch_pubmed_data(self, pmid: str) -> Optional[Dict]:
        """
        Fetch PubMed article data from NCBI.

        Args:
            pmid: PubMed ID

        Returns:
            Dictionary with article data or None if fetch fails
        """
        try:
            time.sleep(self.ncbi_delay)
            url = f"{self.ncbi_base}/efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'xml',
                'email': self.pubmed_email,
                'tool': self.tool_name
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            # Parse XML
            root = ET.fromstring(response.content)
            article = root.find('.//PubmedArticle')

            if article is None:
                print(f"  No article data found for PMID:{pmid}")
                return None

            # Extract article data
            medline = article.find('.//MedlineCitation')
            article_elem = medline.find('.//Article')

            # Title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""

            # Abstract
            abstract_parts = []
            abstract_elem = article_elem.find('.//Abstract')
            if abstract_elem is not None:
                for abstract_text in abstract_elem.findall('.//AbstractText'):
                    if abstract_text.text:
                        label = abstract_text.get('Label', '')
                        text = abstract_text.text
                        if label:
                            abstract_parts.append(f"{label}: {text}")
                        else:
                            abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            # Authors
            authors = []
            author_list = article_elem.find('.//AuthorList')
            if author_list is not None:
                for author in author_list.findall('.//Author'):
                    last_name = author.find('.//LastName')
                    fore_name = author.find('.//ForeName')
                    if last_name is not None and fore_name is not None:
                        authors.append(f"{fore_name.text} {last_name.text}")
                    elif last_name is not None:
                        authors.append(last_name.text)

            # Journal
            journal_elem = article_elem.find('.//Journal')
            journal = ""
            if journal_elem is not None:
                journal_title = journal_elem.find('.//Title')
                if journal_title is not None:
                    journal = journal_title.text

            # Publication date
            pub_date = ""
            pub_date_elem = article_elem.find('.//Journal/JournalIssue/PubDate')
            if pub_date_elem is not None:
                year = pub_date_elem.find('.//Year')
                month = pub_date_elem.find('.//Month')
                day = pub_date_elem.find('.//Day')
                date_parts = []
                if year is not None:
                    date_parts.append(year.text)
                if month is not None:
                    date_parts.append(month.text)
                if day is not None:
                    date_parts.append(day.text)
                pub_date = " ".join(date_parts)

            # DOI
            doi = ""
            article_id_list = article.find('.//PubmedData/ArticleIdList')
            if article_id_list is not None:
                for article_id in article_id_list.findall('.//ArticleId'):
                    if article_id.get('IdType') == 'doi':
                        doi = article_id.text
                        break

            # MeSH terms
            mesh_terms = []
            mesh_heading_list = medline.find('.//MeshHeadingList')
            if mesh_heading_list is not None:
                for mesh_heading in mesh_heading_list.findall('.//MeshHeading'):
                    descriptor = mesh_heading.find('.//DescriptorName')
                    if descriptor is not None:
                        mesh_terms.append(descriptor.text)

            return {
                'pubmed_id': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal,
                'publication_date': pub_date,
                'doi': doi,
                'mesh_terms': mesh_terms
            }

        except Exception as e:
            print(f"  Error fetching PMID:{pmid}: {e}")
            return None

    def log_failed_fetch(self, pmid: str, classification: str, reason: str = "No data returned from NCBI"):
        """
        Log a failed PMID fetch.

        Args:
            pmid: PubMed ID that failed
            classification: Type of paper (seminal_paper, reference, citing_paper)
            reason: Reason for failure
        """
        self.failed_fetches.append({
            'pmid': pmid,
            'classification': classification,
            'reason': reason,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        print(f"  FAILED FETCH: PMID:{pmid} [{classification}] - {reason}")

    def save_failed_fetches_log(self, filename: str = "failed_fetches.json"):
        """Save failed fetches to a JSON log file."""
        if self.failed_fetches:
            with open(filename, 'w') as f:
                json.dump(self.failed_fetches, f, indent=2)
            print(f"\nFailed fetches log saved to {filename}")

    def populate_node_with_logging(self, pmid: str, classification: str) -> bool:
        """
        Fetch PubMed data and populate node with logging.

        Args:
            pmid: PubMed ID
            classification: Type of paper (seminal_paper, reference, citing_paper)

        Returns:
            True if successful, False if data fetch failed
        """
        # Check if already populated
        if self.check_node_exists(pmid):
            print(f"  PMID:{pmid} already exists with data, skipping")
            return True

        # Create empty node first
        self.create_empty_node(pmid)

        # Fetch data from NCBI
        print(f"  Fetching data for PMID:{pmid}")
        data = self.fetch_pubmed_data(pmid)

        if data is None or not data.get('title'):
            self.delete_empty_node(pmid)
            self.log_failed_fetch(pmid, classification)
            return False

        # Generate embedding
        print(f"  Generating embedding for PMID:{pmid}")
        embedding = self.generate_embedding(data['abstract'])

        # Update node with all properties
        with self.driver.session() as session:
            session.run("""
                        MATCH (p:PubMedArticle {pubmed_id: $pmid})
                        SET p.title = $title,
                        p.abstract = $abstract,
                        p.authors = $authors,
                        p.journal = $journal,
                        p.publication_date = $pub_date,
                        p.doi = $doi,
                        p.mesh_terms = $mesh_terms,
                        p.abstract_embedding = $embedding,
                        p.last_updated = datetime()
                        """,
                        pmid=pmid,
                        title=data['title'],
                        abstract=data['abstract'],
                        authors=data['authors'],
                        journal=data['journal'],
                        pub_date=data['publication_date'],
                        doi=data['doi'],
                        mesh_terms=data['mesh_terms'],
                        embedding=embedding
                        )

        print(f"  Successfully populated PMID:{pmid}")
        self.processed_pmids.add(pmid)
        return True

    def create_cites_relationship(self, citing_pmid: str, cited_pmid: str, source: str):
        """
        Create a CITES relationship between two articles.

        Args:
            citing_pmid: PMID of article that cites
            cited_pmid: PMID of article being cited
            source: Source of relationship (e.g., 'seminal_paper', 'reference', 'citing_paper')
        """
        with self.driver.session() as session:
            session.run("""
                        MATCH (citing:PubMedArticle {pubmed_id: $citing_pmid})
                        MATCH (cited:PubMedArticle {pubmed_id: $cited_pmid})
                          WHERE citing.title IS NOT NULL AND cited.title IS NOT NULL
                        MERGE (citing)-[r:CITES]->(cited)
                        SET r.source = $source,
                        r.created_at = CASE WHEN r.created_at IS NULL THEN datetime()
                          ELSE r.created_at
                          END
                        """, citing_pmid=citing_pmid, cited_pmid=cited_pmid, source=source)

    # Citation mining methods (from previous script)
    def get_doi_for_pmid(self, pmid: str) -> Optional[str]:
        """Get DOI for a given PMID."""
        if pmid in self.pmid_to_doi_cache:
            return self.pmid_to_doi_cache[pmid]

        try:
            time.sleep(self.ncbi_delay)
            url = f"{self.ncbi_base}/esummary.fcgi"
            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'json',
                'email': self.pubmed_email,
                'tool': self.tool_name
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'result' in data and pmid in data['result']:
                paper = data['result'][pmid]
                doi = None
                if 'elocationid' in paper:
                    eloc = paper['elocationid']
                    if 'doi:' in eloc.lower():
                        doi = eloc.replace('doi:', '').strip()
                    elif eloc.startswith('10.'):
                        doi = eloc

                if not doi and 'articleids' in paper:
                    for aid in paper['articleids']:
                        if aid.get('idtype') == 'doi':
                            doi = aid.get('value')
                            break

                if doi:
                    self.pmid_to_doi_cache[pmid] = doi
                    return doi

            return None

        except Exception as e:
            print(f"Error fetching DOI for PMID:{pmid}: {e}")
            return None

    def get_pmid_for_doi(self, doi: str) -> Optional[str]:
        """Get PMID for a given DOI."""
        if doi in self.doi_to_pmid_cache:
            return self.doi_to_pmid_cache[doi]

        try:
            time.sleep(self.ncbi_delay)
            url = f"{self.ncbi_base}/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': f'{doi}[doi]',
                'retmode': 'json',
                'email': self.pubmed_email,
                'tool': self.tool_name
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                idlist = data['esearchresult']['idlist']
                if idlist:
                    pmid = idlist[0]
                    self.doi_to_pmid_cache[doi] = pmid
                    return pmid

            return None

        except Exception as e:
            print(f"Error fetching PMID for DOI:{doi}: {e}")
            return None

    def get_references_from_ncbi(self, pmid: str) -> Set[str]:
        """Get references from NCBI E-utilities."""
        url = f"{self.ncbi_base}/elink.fcgi"
        params = {
            'dbfrom': 'pubmed',
            'id': pmid,
            'linkname': 'pubmed_pubmed_refs',
            'retmode': 'json',
            'email': self.pubmed_email,
            'tool': self.tool_name
        }

        try:
            time.sleep(self.ncbi_delay)
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            references = set()
            if 'linksets' in data and len(data['linksets']) > 0:
                linkset = data['linksets'][0]
                if 'linksetdbs' in linkset:
                    for linksetdb in linkset['linksetdbs']:
                        if 'links' in linksetdb:
                            references.update(linksetdb['links'])

            return references

        except Exception as e:
            print(f"Error fetching NCBI references for PMID:{pmid}: {e}")
            return set()

    def get_references_from_crossref(self, doi: str) -> Set[str]:
        """Get references from Crossref and map to PMIDs."""
        try:
            time.sleep(self.crossref_delay)
            url = f"{self.crossref_base}/{doi}"
            headers = {'User-Agent': f'{self.tool_name} (mailto:{self.pubmed_email})'}

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            reference_pmids = set()

            if 'message' in data and 'reference' in data['message']:
                references = data['message']['reference']
                for ref in references:
                    ref_doi = ref.get('DOI')
                    if ref_doi:
                        pmid = self.get_pmid_for_doi(ref_doi)
                        if pmid:
                            reference_pmids.add(pmid)

            return reference_pmids

        except Exception as e:
            print(f"Error fetching Crossref references for DOI:{doi}: {e}")
            return set()

    def get_references(self, pmid: str) -> Set[str]:
        """Get all references using both NCBI and Crossref."""
        print(f"  Fetching references from NCBI...")
        ncbi_refs = self.get_references_from_ncbi(pmid)
        print(f"  Found {len(ncbi_refs)} NCBI references")

        doi = self.get_doi_for_pmid(pmid)
        crossref_refs = set()

        if doi:
            print(f"  Fetching references from Crossref...")
            crossref_refs = self.get_references_from_crossref(doi)
            print(f"  Found {len(crossref_refs)} Crossref references")

        all_refs = ncbi_refs | crossref_refs
        print(f"  Total unique references: {len(all_refs)}")
        return all_refs

    def get_citing_papers(self, pmid: str, max_results: int = 500) -> Set[str]:
        """Get papers that cite the input paper using Europe PMC."""
        citing_pmids = set()
        page_size = 100
        cursor_mark = "*"

        try:
            while len(citing_pmids) < max_results:
                time.sleep(self.europepmc_delay)

                url = f"{self.europepmc_base}/search"
                params = {
                    'query': f'CITES:{pmid}_MED',
                    'format': 'json',
                    'pageSize': page_size,
                    'cursorMark': cursor_mark
                }

                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if 'resultList' not in data or 'result' not in data['resultList']:
                    break

                results = data['resultList']['result']
                if not results:
                    break

                for result in results:
                    if 'pmid' in result:
                        citing_pmids.add(result['pmid'])

                next_cursor = data.get('nextCursorMark')
                if not next_cursor or next_cursor == cursor_mark:
                    break
                cursor_mark = next_cursor

                if len(results) < page_size:
                    break

            print(f"  Found {len(citing_pmids)} citing papers")
            return citing_pmids

        except Exception as e:
            print(f"Error fetching citing papers for PMID:{pmid}: {e}")
            return citing_pmids

    def build_knowledge_graph(
            self,
            seminal_pmids: List[str],
            max_citing_papers: int = 1000,
            process_references: bool = True,
            process_citing: bool = True
    ):
        """
        Build complete knowledge graph in Neo4j from seminal papers.

        Args:
            seminal_pmids: List of seminal paper PMIDs
            max_citing_papers: Maximum citing papers per seminal paper
            process_references: Whether to process reference papers
            process_citing: Whether to process citing papers
        """
        print("=" * 70)
        print("CANCER KNOWLEDGE GRAPH BUILDER")
        print("=" * 70)

        stats = {
            'seminal_processed': 0,
            'references_processed': 0,
            'citing_processed': 0,
            'nodes_created': 0,
            'relationships_created': 0,
            'failed_fetches': 0
        }

        for seminal_pmid in seminal_pmids:
            print(f"\n{'=' * 70}")
            print(f"PROCESSING SEMINAL PAPER: PMID:{seminal_pmid}")
            print(f"{'=' * 70}")

            # Process seminal paper node
            if self.populate_node_with_logging(seminal_pmid, 'seminal_paper'):
                stats['seminal_processed'] += 1
                stats['nodes_created'] += 1
            else:
                print(f"WARNING: Failed to fetch seminal paper PMID:{seminal_pmid}")
                stats['failed_fetches'] += 1
                continue

            # Process references
            if process_references:
                print(f"\n--- Processing References for PMID:{seminal_pmid} ---")
                references = self.get_references(seminal_pmid)

                for i, ref_pmid in enumerate(references, 1):
                    print(f"\nReference {i}/{len(references)}: PMID:{ref_pmid}")

                    if self.populate_node_with_logging(ref_pmid, 'reference'):
                        stats['references_processed'] += 1
                        stats['nodes_created'] += 1

                        # Create CITES relationship: seminal -> reference
                        self.create_cites_relationship(seminal_pmid, ref_pmid, 'seminal_paper')
                        stats['relationships_created'] += 1
                        print(f"  Created: ({seminal_pmid})-[:CITES {{source:'seminal_paper'}}]->({ref_pmid})")
                    else:
                        stats['failed_fetches'] += 1

            # Process citing papers
            if process_citing:
                print(f"\n--- Processing Citing Papers for PMID:{seminal_pmid} ---")
                citing_papers = self.get_citing_papers(seminal_pmid, max_results=max_citing_papers)

                for i, citing_pmid in enumerate(citing_papers, 1):
                    print(f"\nCiting Paper {i}/{len(citing_papers)}: PMID:{citing_pmid}")

                    if self.populate_node_with_logging(citing_pmid, 'citing_paper'):
                        stats['citing_processed'] += 1
                        stats['nodes_created'] += 1

                        # Create CITES relationship: citing -> seminal
                        self.create_cites_relationship(citing_pmid, seminal_pmid, 'citing_paper')
                        stats['relationships_created'] += 1
                        print(f"  Created: ({citing_pmid})-[:CITES {{source:'citing_paper'}}]->({seminal_pmid})")
                    else:
                        stats['failed_fetches'] += 1

        # Print final statistics
        print(f"\n{'=' * 70}")
        print("KNOWLEDGE GRAPH BUILD COMPLETE")
        print(f"{'=' * 70}")
        print(f"Seminal papers processed: {stats['seminal_processed']}")
        print(f"References processed: {stats['references_processed']}")
        print(f"Citing papers processed: {stats['citing_processed']}")
        print(f"Total nodes created: {stats['nodes_created']}")
        print(f"Total relationships created: {stats['relationships_created']}")
        print(f"Failed fetches: {stats['failed_fetches']}")
        print(f"{'=' * 70}")

        # Save failed fetches log
        if self.failed_fetches:
            self.save_failed_fetches_log()
            print(f"\nWARNING: {len(self.failed_fetches)} PMIDs failed to fetch. See failed_fetches.json for details.")


# Example usage
if __name__ == "__main__":
    # Load environment variables
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    PUBMED_EMAIL = os.getenv('NCBI_EMAIL')

    # Check number of arguments (excluding script name)
    num_args = len(sys.argv) - 1

    if num_args < 1 or num_args > 10:
        print(f"Error: Expected 1 to 10 arguments, but received {num_args}")
        print("Usage: python script.py <int1> [int2] [int3] [int4] [int5] [int6]")
        sys.exit(1)
        # Validate that each argument is an integer string
    seminal_papers = []
    for i, arg in enumerate(sys.argv[1:], start=1):
        try:
        # Convert to integer (will raise ValueError if not a valid integer)
            num = int(arg)
            seminal_papers.append(arg)
        except ValueError:
            print(f"Error: Argument {i} ('{arg}') is not a valid integer string")
            sys.exit(1)


    if not NEO4J_PASSWORD or not PUBMED_EMAIL:
        raise ValueError("Please set NEO4J_PASSWORD and NCBI_EMAIL environment variables")

    # Initialize builder
    builder = Neo4jCancerKGBuilder(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        pubmed_email=PUBMED_EMAIL
    )

    try:
        # Seminal cancer genomics papers
       # seminal_papers = [
            #"21376230",  # Hallmarks of Cancer: The Next Generation
           # "11381259",  # Hallmarks of Cancer (original)
           # "23539594",  # The Cancer Genome Atlas (TCGA)
           #"23917401",  # COSMIC database paper
           # "23540688",  # Lessons from the Cancer Genome
        #]

        # Build the knowledge graph
        builder.build_knowledge_graph(
            seminal_pmids=seminal_papers,
            max_citing_papers=200,  # Limit for demonstration
            process_references=True,
            process_citing=True
        )

    finally:
        builder.close()
        print("\nNeo4j connection closed")