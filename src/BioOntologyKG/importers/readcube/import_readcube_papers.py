#!/usr/bin/env python3
"""
Import ReadCube Papers metadata from .bib file into Neo4j PubMedArticle nodes.
"""

import os
import re
from typing import Dict, Optional
from neo4j import GraphDatabase
import bibtexparser
from bibtexparser.bparser import BibTexParser


class Neo4jPapersImporter:
    def __init__(self):
        """Initialize Neo4j connection using environment variables."""
        self.uri = os.getenv('NEO4J_URI')
        self.user = os.getenv('NEO4J_USERNAME')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.database = os.getenv('NEO4J_DATABASE')

        if not all([self.uri, self.user, self.password, self.database]):
            raise ValueError(
                "Missing required environment variables: "
                "NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE"
            )

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )

    def close(self):
        """Close Neo4j driver connection."""
        self.driver.close()

    def clean_braces(self, text: str) -> str:
        """Remove all brace characters from text."""
        if not text:
            return text
        return text.replace('{{', '').replace('}}', '').replace('{', '').replace('}', '')

    def parse_authors(self, author_string: str) -> list:
        """Parse author string into a list."""
        if not author_string:
            return []
        # Split by ' and ' to separate multiple authors
        authors = [self.clean_braces(a.strip()) for a in author_string.split(' and ')]
        return [a for a in authors if a]  # Remove empty strings

    def create_pmcid_url(self, pmcid: str) -> Optional[str]:
        """Convert PMCID to full URL."""
        if not pmcid:
            return None
        clean_pmcid = self.clean_braces(pmcid).strip()
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{clean_pmcid}"

    def is_curated(self, entry: Dict) -> bool:
        """Check if article is curated (not 'undefined')."""
        entry_id = entry.get('ID', '')
        return entry_id != 'undefined'

    def check_if_already_imported(self, pubmed_id: str) -> bool:
        """Check if article already has a local-url in Neo4j."""
        query = """
        MATCH (p:PubMedArticle {pubmed_id: $pubmed_id})
        WHERE p.`local-url` IS NOT NULL
        RETURN count(p) > 0 as exists
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, pubmed_id=pubmed_id)
            record = result.single()
            return record['exists'] if record else False

    def import_article(self, entry: Dict):
        """Import or update a single article in Neo4j."""
        # Extract and clean required fields
        pmid = self.clean_braces(entry.get('pmid', '')).strip()

        if not pmid:
            print(f"Skipping article without PMID: {entry.get('title', 'Unknown')[:50]}...")
            return

        # Check if already imported
        if self.check_if_already_imported(pmid):
            print(f"Skipping already imported article: PMID {pmid}")
            return

        # Prepare properties
        properties = {
            'pubmed_id': pmid,
            'publication_date': self.clean_braces(entry.get('year', '')).strip(),
            'title': self.clean_braces(entry.get('title', '')).strip(),
            'authors': self.parse_authors(entry.get('author', '')),
            'journal': self.clean_braces(entry.get('journal', '')).strip(),
            'doi': self.clean_braces(entry.get('doi', '')).strip(),
            'abstract': self.clean_braces(entry.get('abstract', '')).strip(),
            'pages': self.clean_braces(entry.get('pages', '')).strip(),
            'number': self.clean_braces(entry.get('number', '')).strip(),
            'volume': self.clean_braces(entry.get('volume', '')).strip(),
            'local-url': self.clean_braces(entry.get('local-url', '')).strip()
        }

        # Add PMCID URL if available
        pmcid = entry.get('pmcid', '')
        if pmcid:
            properties['pmcid_url'] = self.create_pmcid_url(pmcid)

        # Remove empty string values
        properties = {k: v for k, v in properties.items() if v or isinstance(v, list)}

        # Create or update node in Neo4j
        query = """
        MERGE (p:PubMedArticle {pubmed_id: $pubmed_id})
        SET p += $properties
        RETURN p.pubmed_id as pubmed_id
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query, pubmed_id=pmid, properties=properties)
            record = result.single()
            if record:
                print(f"✓ Imported/Updated: PMID {record['pubmed_id']} - {properties.get('title', '')[:60]}...")

    def process_bib_file(self, filepath: str):
        """Process the .bib file and import articles."""
        print(f"\nReading .bib file: {filepath}")

        # Configure parser to handle common BibTeX variations
        parser = BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False

        with open(filepath, 'r', encoding='utf-8') as bibfile:
            bib_database = bibtexparser.load(bibfile, parser)

        total_entries = len(bib_database.entries)
        print(f"Found {total_entries} entries in .bib file\n")

        processed = 0
        skipped_uncurated = 0
        skipped_no_pmid = 0
        skipped_already_imported = 0
        imported = 0

        for entry in bib_database.entries:
            # Only process @article entries
            if entry.get('ENTRYTYPE') != 'article':
                continue

            processed += 1

            # Check if curated
            if not self.is_curated(entry):
                skipped_uncurated += 1
                print(f"Skipping uncurated article: {entry.get('title', 'Unknown')[:50]}...")
                continue

            # Check for PMID
            if not entry.get('pmid'):
                skipped_no_pmid += 1
                continue

            try:
                pmid = self.clean_braces(entry.get('pmid', '')).strip()
                if self.check_if_already_imported(pmid):
                    skipped_already_imported += 1
                    continue

                self.import_article(entry)
                imported += 1
            except Exception as e:
                print(f"✗ Error processing article PMID {entry.get('pmid', 'Unknown')}: {str(e)}")

        # Print summary
        print("\n" + "=" * 70)
        print("IMPORT SUMMARY")
        print("=" * 70)
        print(f"Total entries processed: {processed}")
        print(f"Articles imported/updated: {imported}")
        print(f"Skipped (uncurated): {skipped_uncurated}")
        print(f"Skipped (no PMID): {skipped_no_pmid}")
        print(f"Skipped (already imported): {skipped_already_imported}")
        print("=" * 70)


def main():
    """Main function to run the importer."""
    print("=" * 70)
    print("ReadCube Papers to Neo4j Importer")
    print("=" * 70)

    # Prompt for .bib file location
    bib_filepath = input("\nEnter the path to your .bib file: ").strip()

    # Remove quotes if user wrapped path in quotes
    bib_filepath = bib_filepath.strip('"').strip("'")

    if not os.path.exists(bib_filepath):
        print(f"Error: File not found: {bib_filepath}")
        return

    # Initialize importer
    try:
        importer = Neo4jPapersImporter()
        print("✓ Connected to Neo4j database")

        # Process the file
        importer.process_bib_file(bib_filepath)

        # Clean up
        importer.close()
        print("\n✓ Import complete. Connection closed.")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()