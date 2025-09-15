#!/usr/bin/env python3
"""
PubMed Data Fetcher using BioPython
Fetches full PubMed entries and maps them to structured fields
"""

from Bio import Entrez
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional
import json

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
                print(f"No article found for PMID: {pubmed_id}")
                return None
                
            article = records['PubmedArticle'][0]
            return self._parse_article(article)
            
        except Exception as e:
            print(f"Error fetching PMID {pubmed_id}: {str(e)}")
            return None
    
    def _parse_article(self, article: Dict) -> Dict:
        """Parse the raw article data into structured fields"""
        
        medline_citation = article['MedlineCitation']
        pubmed_data = article.get('PubmedData', {})
        
        # Basic article info
        article_data = medline_citation['Article']
        
        # Parse structured data
        parsed_data = {
            'pmid': medline_citation['PMID'],
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
            'status': medline_citation.get('Status', ''),
            'date_created': self._get_date_created(medline_citation),
            'date_revised': self._get_date_revised(medline_citation)
        }
        
        return parsed_data
    
    def _get_title(self, article_data: Dict) -> str:
        """Extract article title"""
        return article_data.get('ArticleTitle', '')
    
    def _get_abstract(self, article_data: Dict) -> str:
        """Extract and concatenate abstract text"""
        abstract = article_data.get('Abstract', {})
        if not abstract:
            return ''
            
        abstract_texts = abstract.get('AbstractText', [])
        if isinstance(abstract_texts, str):
            return abstract_texts
        elif isinstance(abstract_texts, list):
            # Handle structured abstracts with labels
            full_abstract = []
            for text_section in abstract_texts:
                if hasattr(text_section, 'attributes') and 'Label' in text_section.attributes:
                    label = text_section.attributes['Label']
                    full_abstract.append(f"{label}: {str(text_section)}")
                else:
                    full_abstract.append(str(text_section))
            return ' '.join(full_abstract)
        return ''
    
    def _get_authors(self, article_data: Dict) -> List[Dict]:
        """Extract author information"""
        authors = []
        author_list = article_data.get('AuthorList', [])
        
        for author in author_list:
            author_info = {
                'last_name': author.get('LastName', ''),
                'first_name': author.get('ForeName', ''),
                'initials': author.get('Initials', ''),
                'affiliation': author.get('AffiliationInfo', [{}])[0].get('Affiliation', '') if author.get('AffiliationInfo') else '',
                'orcid': self._extract_orcid(author)
            }
            authors.append(author_info)
        
        return authors
    
    def _extract_orcid(self, author: Dict) -> str:
        """Extract ORCID from author data"""
        identifiers = author.get('Identifier', [])
        for identifier in identifiers:
            if identifier.get('Source') == 'ORCID':
                return identifier.get('content', '')
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
        year = pub_date.get('Year', '')
        month = pub_date.get('Month', '')
        day = pub_date.get('Day', '')
        
        date_parts = [part for part in [year, month, day] if part]
        return ' '.join(date_parts)
    
    def _get_publication_date(self, article_data: Dict) -> str:
        """Extract publication date"""
        article_date = article_data.get('ArticleDate', [])
        if article_date:
            date = article_date[0]
            year = date.get('Year', '')
            month = date.get('Month', '').zfill(2)
            day = date.get('Day', '').zfill(2)
            return f"{year}-{month}-{day}" if all([year, month, day]) else year
        return ''
    
    def _get_doi(self, article_data: Dict) -> str:
        """Extract DOI"""
        elocation_id = article_data.get('ELocationID', [])
        for location in elocation_id:
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
                'major_topic': descriptor.attributes.get('MajorTopicYN') == 'Y',
                'qualifiers': [
                    {
                        'qualifier': str(qual),
                        'major_topic': qual.attributes.get('MajorTopicYN') == 'Y'
                    } for qual in qualifiers
                ]
            }
            mesh_terms.append(mesh_term)
        
        return mesh_terms
    
    def _get_publication_types(self, article_data: Dict) -> List[str]:
        """Extract publication types"""
        pub_type_list = article_data.get('PublicationTypeList', [])
        return [str(pub_type) for pub_type in pub_type_list]
    
    def _get_language(self, article_data: Dict) -> List[str]:
        """Extract language information"""
        return article_data.get('Language', [])
    
    def _get_country(self, medline_citation: Dict) -> str:
        """Extract country of publication"""
        journal_info = medline_citation.get('Article', {}).get('Journal', {}).get('JournalIssue', {})
        medline_journal_info = medline_citation.get('MedlineJournalInfo', {})
        return medline_journal_info.get('Country', '')
    
    def _get_affiliations(self, article_data: Dict) -> List[str]:
        """Extract all unique affiliations"""
        affiliations = set()
        author_list = article_data.get('AuthorList', [])
        
        for author in author_list:
            affiliation_info = author.get('AffiliationInfo', [])
            for affiliation in affiliation_info:
                if affiliation.get('Affiliation'):
                    affiliations.add(affiliation['Affiliation'])
        
        return list(affiliations)
    
    def _get_grants(self, article_data: Dict) -> List[Dict]:
        """Extract grant information"""
        grant_list = article_data.get('GrantList', [])
        grants = []
        
        for grant in grant_list:
            grant_info = {
                'grant_id': grant.get('GrantID', ''),
                'acronym': grant.get('Acronym', ''),
                'agency': grant.get('Agency', ''),
                'country': grant.get('Country', '')
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
        
        year = date_dict.get('Year', '')
        month = date_dict.get('Month', '').zfill(2)
        day = date_dict.get('Day', '').zfill(2)
        
        return f"{year}-{month}-{day}" if all([year, month, day]) else year


def main():
    """Example usage"""
    # Initialize fetcher with your email
    fetcher = PubMedFetcher("your.email@example.com")  # Replace with your email
    
    # Example PMIDs for testing
    test_pmids = ["33747033", "32073213", "31586400"]  # Replace with PMIDs of interest
    
    for pmid in test_pmids:
        print(f"\n{'='*50}")
        print(f"Fetching PMID: {pmid}")
        print('='*50)
        
        article_data = fetcher.fetch_pubmed_article(pmid)
        
        if article_data:
            # Print key fields
            print(f"Title: {article_data['title']}")
            print(f"Authors: {len(article_data['authors'])} authors")
            if article_data['authors']:
                first_author = article_data['authors'][0]
                print(f"First author: {first_author['first_name']} {first_author['last_name']}")
            
            print(f"Journal: {article_data['journal']['title']}")
            print(f"Publication Date: {article_data['publication_date']}")
            print(f"DOI: {article_data['doi']}")
            print(f"Abstract length: {len(article_data['abstract'])} characters")
            print(f"Keywords: {len(article_data['keywords'])} keywords")
            print(f"MeSH Terms: {len(article_data['mesh_terms'])} terms")
            
            # Optionally save to JSON file
            with open(f"pubmed_{pmid}.json", 'w') as f:
                json.dump(article_data, f, indent=2, default=str)
            print(f"Saved data to pubmed_{pmid}.json")


if __name__ == "__main__":
    main()