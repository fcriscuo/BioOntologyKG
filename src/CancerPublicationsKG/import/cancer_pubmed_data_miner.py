"""
Cancer Genomics Knowledge Graph Miner
Mines PubMed citations from seminal papers using NCBI E-utilities, Europe PMC, and Crossref APIs
"""

import requests
import time
import json
from typing import Set, Dict, List, Optional, Tuple
from collections import defaultdict


class PubMedCitationMiner:
    def __init__(self, email: str, tool_name: str = "CancerKnowledgeGraphMiner"):
        """
        Initialize the citation miner.

        Args:
            email: Your email (required by NCBI for E-utilities and Crossref)
            tool_name: Name of your application
        """
        self.email = email
        self.tool_name = tool_name
        self.ncbi_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.europepmc_base = "https://www.ebi.ac.uk/europepmc/webservices/rest"
        self.icite_base = "https://icite.od.nih.gov/api/pubs"
        self.crossref_base = "https://api.crossref.org/works"

        # Rate limiting
        self.ncbi_delay = 0.34  # ~3 requests per second (conservative)
        self.europepmc_delay = 0.2
        self.crossref_delay = 1.0  # Be polite to Crossref

        # Cache for DOI to PMID mappings
        self.doi_to_pmid_cache = {}
        self.pmid_to_doi_cache = {}

    def get_doi_for_pmid(self, pmid: str) -> Optional[str]:
        """
        Get DOI for a given PMID using NCBI E-utilities.

        Args:
            pmid: PubMed ID

        Returns:
            DOI string or None
        """
        if pmid in self.pmid_to_doi_cache:
            return self.pmid_to_doi_cache[pmid]

        try:
            time.sleep(self.ncbi_delay)
            url = f"{self.ncbi_base}/esummary.fcgi"
            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'json',
                'email': self.email,
                'tool': self.tool_name
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'result' in data and pmid in data['result']:
                paper = data['result'][pmid]
                # Try multiple DOI fields
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
        """
        Get PMID for a given DOI using NCBI E-utilities.

        Args:
            doi: DOI string

        Returns:
            PMID string or None
        """
        if doi in self.doi_to_pmid_cache:
            return self.doi_to_pmid_cache[doi]

        try:
            time.sleep(self.ncbi_delay)
            url = f"{self.ncbi_base}/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': f'{doi}[doi]',
                'retmode': 'json',
                'email': self.email,
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
        """
        Get references from NCBI E-utilities (returns PMIDs only).

        Args:
            pmid: PubMed ID

        Returns:
            Set of PMIDs that are referenced by the input paper
        """
        url = f"{self.ncbi_base}/elink.fcgi"
        params = {
            'dbfrom': 'pubmed',
            'id': pmid,
            'linkname': 'pubmed_pubmed_refs',
            'retmode': 'json',
            'email': self.email,
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

    def get_references_from_crossref(self, doi: str) -> Tuple[Set[str], List[Dict]]:
        """
        Get references from Crossref API (more comprehensive than NCBI).

        Args:
            doi: DOI of the paper

        Returns:
            Tuple of (Set of PMIDs found, List of all reference metadata)
        """
        try:
            time.sleep(self.crossref_delay)
            url = f"{self.crossref_base}/{doi}"
            headers = {
                'User-Agent': f'{self.tool_name} (mailto:{self.email})'
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            references_metadata = []
            reference_pmids = set()

            if 'message' in data and 'reference' in data['message']:
                references = data['message']['reference']
                print(f"  Crossref found {len(references)} total references")

                for ref in references:
                    ref_info = {
                        'doi': ref.get('DOI'),
                        'title': ref.get('article-title'),
                        'author': ref.get('author'),
                        'year': ref.get('year'),
                        'journal': ref.get('journal-title'),
                        'volume': ref.get('volume'),
                        'page': ref.get('first-page')
                    }
                    references_metadata.append(ref_info)

                    # Try to get PMID for this reference
                    ref_doi = ref.get('DOI')
                    if ref_doi:
                        pmid = self.get_pmid_for_doi(ref_doi)
                        if pmid:
                            reference_pmids.add(pmid)

                print(f"  Mapped {len(reference_pmids)} references to PMIDs")

            return reference_pmids, references_metadata

        except Exception as e:
            print(f"Error fetching Crossref references for DOI:{doi}: {e}")
            return set(), []

    def get_references(self, pmid: str) -> Tuple[Set[str], Dict]:
        """
        Get all references (backward citations) using both NCBI and Crossref.

        Args:
            pmid: PubMed ID

        Returns:
            Tuple of (Set of PMIDs, metadata dictionary)
        """
        print(f"  Fetching references from NCBI...")
        ncbi_refs = self.get_references_from_ncbi(pmid)
        print(f"  NCBI found {len(ncbi_refs)} references with PMIDs")

        # Try Crossref for more comprehensive data
        doi = self.get_doi_for_pmid(pmid)
        crossref_refs = set()
        crossref_metadata = []

        if doi:
            print(f"  Fetching references from Crossref (DOI: {doi})...")
            crossref_refs, crossref_metadata = self.get_references_from_crossref(doi)
        else:
            print(f"  No DOI found, skipping Crossref lookup")

        # Combine results
        all_refs = ncbi_refs | crossref_refs

        metadata = {
            'ncbi_count': len(ncbi_refs),
            'crossref_count': len(crossref_refs),
            'total_unique': len(all_refs),
            'crossref_metadata': crossref_metadata,
            'doi': doi
        }

        print(f"  Combined total: {len(all_refs)} unique references with PMIDs")
        return all_refs, metadata

    def get_citing_papers(self, pmid: str, max_results: int = 1000) -> Set[str]:
        """
        Get papers that cite the input paper (forward citations) using Europe PMC.

        Args:
            pmid: PubMed ID
            max_results: Maximum number of citing papers to retrieve

        Returns:
            Set of PMIDs that cite the input paper
        """
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

                # Check if there are more results
                next_cursor = data.get('nextCursorMark')
                if not next_cursor or next_cursor == cursor_mark:
                    break
                cursor_mark = next_cursor

                if len(results) < page_size:
                    break

            print(f"Found {len(citing_pmids)} citing papers for PMID:{pmid}")
            return citing_pmids

        except Exception as e:
            print(f"Error fetching citing papers for PMID:{pmid}: {e}")
            return citing_pmids

    def get_citation_metrics(self, pmids: List[str]) -> Dict[str, Dict]:
        """
        Get citation metrics from NIH iCite for a list of PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            Dictionary mapping PMIDs to their citation metrics
        """
        metrics = {}
        batch_size = 1000

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            pmid_string = ','.join(batch)

            try:
                time.sleep(0.2)
                url = f"{self.icite_base}"
                params = {
                    'pmids': pmid_string,
                    'fl': 'pmid,year,citation_count,relative_citation_ratio,nih_percentile,cited_by_clin'
                }

                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if 'data' in data:
                    for paper in data['data']:
                        pmid = str(paper.get('pmid'))
                        metrics[pmid] = {
                            'citation_count': paper.get('citation_count', 0),
                            'year': paper.get('year'),
                            'rcr': paper.get('relative_citation_ratio'),
                            'nih_percentile': paper.get('nih_percentile'),
                            'cited_by_clin': paper.get('cited_by_clin')
                        }

            except Exception as e:
                print(f"Error fetching metrics for batch: {e}")

        print(f"Retrieved metrics for {len(metrics)} papers")
        return metrics

    def get_paper_details(self, pmids: List[str]) -> Dict[str, Dict]:
        """
        Get detailed information for papers using NCBI E-utilities.

        Args:
            pmids: List of PubMed IDs

        Returns:
            Dictionary mapping PMIDs to paper details
        """
        details = {}
        batch_size = 200

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            pmid_string = ','.join(batch)

            try:
                time.sleep(self.ncbi_delay)
                url = f"{self.ncbi_base}/esummary.fcgi"
                params = {
                    'db': 'pubmed',
                    'id': pmid_string,
                    'retmode': 'json',
                    'email': self.email,
                    'tool': self.tool_name
                }

                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if 'result' in data:
                    for pmid in batch:
                        if pmid in data['result']:
                            paper = data['result'][pmid]
                            details[pmid] = {
                                'title': paper.get('title', ''),
                                'authors': [author.get('name', '') for author in paper.get('authors', [])],
                                'journal': paper.get('fulljournalname', ''),
                                'pubdate': paper.get('pubdate', ''),
                                'doi': paper.get('elocationid', '')
                            }

            except Exception as e:
                print(f"Error fetching details for batch: {e}")

        return details

    def mine_knowledge_graph(
            self,
            seminal_pmids: List[str],
            min_citations: int = 50,
            max_citing_papers: int = 500
    ) -> Dict:
        """
        Mine a knowledge graph from seminal papers.

        Args:
            seminal_pmids: List of seminal paper PMIDs to start from
            min_citations: Minimum citations for a paper to be included
            max_citing_papers: Maximum citing papers to retrieve per seminal paper

        Returns:
            Dictionary containing the knowledge graph structure
        """
        print("=" * 60)
        print("Cancer Genomics Knowledge Graph Mining")
        print("=" * 60)

        graph = {
            'seminal_papers': {},
            'referenced_papers': defaultdict(lambda: {
                'referenced_by': [],
                'citations': 0,
                'has_crossref_data': False
            }),
            'citing_papers': defaultdict(lambda: {'cites': [], 'citations': 0}),
            'metadata': {},
            'reference_coverage': {}
        }

        all_pmids = set()

        # Step 1: Process each seminal paper
        for seminal_pmid in seminal_pmids:
            print(f"\n--- Processing Seminal Paper: PMID:{seminal_pmid} ---")

            # Get references (backward citations) from both NCBI and Crossref
            references, ref_metadata = self.get_references(seminal_pmid)

            graph['seminal_papers'][seminal_pmid] = {
                'references': list(references),
                'citing_papers': [],
                'reference_metadata': ref_metadata
            }

            # Track coverage statistics
            graph['reference_coverage'][seminal_pmid] = {
                'ncbi_refs': ref_metadata['ncbi_count'],
                'crossref_total': len(ref_metadata['crossref_metadata']),
                'crossref_with_pmid': ref_metadata['crossref_count'],
                'combined_pmids': ref_metadata['total_unique']
            }

            for ref_pmid in references:
                graph['referenced_papers'][ref_pmid]['referenced_by'].append(seminal_pmid)
                if ref_metadata['crossref_count'] > 0:
                    graph['referenced_papers'][ref_pmid]['has_crossref_data'] = True

            all_pmids.update(references)

            # Get citing papers (forward citations)
            print(f"  Fetching citing papers...")
            citing = self.get_citing_papers(seminal_pmid, max_results=max_citing_papers)
            graph['seminal_papers'][seminal_pmid]['citing_papers'] = list(citing)

            for citing_pmid in citing:
                graph['citing_papers'][citing_pmid]['cites'].append(seminal_pmid)

            all_pmids.update(citing)

        # Step 2: Get citation metrics for all papers
        print(f"\n--- Fetching citation metrics for {len(all_pmids)} papers ---")
        metrics = self.get_citation_metrics(list(all_pmids))

        # Update citation counts
        for pmid, metric_data in metrics.items():
            if pmid in graph['referenced_papers']:
                graph['referenced_papers'][pmid]['citations'] = metric_data['citation_count']
            if pmid in graph['citing_papers']:
                graph['citing_papers'][pmid]['citations'] = metric_data['citation_count']

        # Step 3: Filter by citation threshold
        print(f"\n--- Filtering papers with >= {min_citations} citations ---")
        graph['referenced_papers'] = {
            pmid: data for pmid, data in graph['referenced_papers'].items()
            if data['citations'] >= min_citations
        }
        graph['citing_papers'] = {
            pmid: data for pmid, data in graph['citing_papers'].items()
            if data['citations'] >= min_citations
        }

        # Step 4: Get paper details for high-value papers
        high_value_pmids = (
                list(graph['referenced_papers'].keys())[:100] +
                list(graph['citing_papers'].keys())[:100]
        )
        print(f"\n--- Fetching details for top {len(high_value_pmids)} papers ---")
        graph['metadata'] = self.get_paper_details(high_value_pmids)

        # Summary statistics
        print("\n" + "=" * 60)
        print("MINING COMPLETE - Summary Statistics:")
        print("=" * 60)
        print(f"Seminal papers: {len(seminal_pmids)}")
        print(f"Referenced papers (>= {min_citations} citations): {len(graph['referenced_papers'])}")
        print(f"Citing papers (>= {min_citations} citations): {len(graph['citing_papers'])}")
        print(f"Total unique papers in graph: {len(all_pmids)}")
        print(f"Papers with metadata: {len(graph['metadata'])}")

        print("\nReference Coverage Analysis:")
        for pmid, coverage in graph['reference_coverage'].items():
            print(f"  PMID:{pmid}")
            print(f"    NCBI refs with PMIDs: {coverage['ncbi_refs']}")
            print(f"    Crossref total refs: {coverage['crossref_total']}")
            print(f"    Crossref refs with PMIDs: {coverage['crossref_with_pmid']}")
            print(f"    Combined unique PMIDs: {coverage['combined_pmids']}")
            if coverage['crossref_total'] > 0:
                pmid_rate = (coverage['combined_pmids'] / coverage['crossref_total']) * 100
                print(f"    PMID mapping rate: {pmid_rate:.1f}%")

        return graph

    def export_graph(self, graph: Dict, filename: str = "cancer_knowledge_graph.json"):
        """Export the knowledge graph to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(graph, f, indent=2)
        print(f"\nKnowledge graph exported to {filename}")


# Example usage
if __name__ == "__main__":
    # Initialize the miner with your email
    miner = PubMedCitationMiner(email="fcriscuo@genomicsai.dev")

    # Seminal cancer genomics papers
    seminal_papers = [
        "21376230",  # Hallmarks of Cancer: The Next Generation
        "11381259",  # Hallmarks of Cancer (original)
        "23539594",  # The Cancer Genome Atlas (TCGA)
        "23917401",  # COSMIC database paper
        "23540688", # Lessons from the Cancer Genome
    ]

    # Mine the knowledge graph
    graph = miner.mine_knowledge_graph(
        seminal_pmids=seminal_papers,
        min_citations=50,  # Only include papers with >= 50 citations
        max_citing_papers=2000  # Get up to 500 citing papers per seminal paper
    )

    # Export to JSON
    miner.export_graph(graph, "cancer_genomics_kg.json")

    # Example: Print top 10 most cited referenced papers
    print("\n" + "=" * 60)
    print("Top 10 Most Cited Referenced Papers:")
    print("=" * 60)
    referenced_sorted = sorted(
        graph['referenced_papers'].items(),
        key=lambda x: x[1]['citations'],
        reverse=True
    )[:10]

    for pmid, data in referenced_sorted:
        title = graph['metadata'].get(pmid, {}).get('title', 'N/A')
        crossref_flag = " [Crossref]" if data.get('has_crossref_data') else ""
        print(f"PMID:{pmid} - {data['citations']} citations{crossref_flag}")
        print(f"  Title: {title[:100]}...")
        print()