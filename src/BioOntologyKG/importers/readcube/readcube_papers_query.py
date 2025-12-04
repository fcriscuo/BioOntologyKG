#!/usr/bin/env python3
"""
ReadCube Papers Repository Query Script

This script queries a Neo4j database to determine if scientific papers
are available in a local ReadCube Papers repository.
"""

import os
import sys
from neo4j import GraphDatabase


class ReadCubeQueryTool:
    """Tool for querying ReadCube Papers repository in Neo4j database."""

    def __init__(self, uri, user, password, database):
        """Initialize Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.database = database
            # Test connection
            self.driver.verify_connectivity()
            print("✓ Successfully connected to Neo4j database\n")
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {e}")
            sys.exit(1)

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

    def query_by_pubmed_id(self, pubmed_id):
        """Query by PubMed ID."""
        query = """
        MATCH (a:PubMedArticle)
        WHERE a.pubmed_id = $pubmed_id 
          AND a.`local-url` IS NOT NULL
        RETURN a.pubmed_id AS pubmed_id,
               a.journal AS journal,
               a.title AS title
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, pubmed_id=pubmed_id)
            return list(result)

    def query_by_doi(self, doi):
        """Query by DOI."""
        query = """
        MATCH (a:PubMedArticle)
        WHERE a.doi = $doi 
          AND a.`local-url` IS NOT NULL
        RETURN a.pubmed_id AS pubmed_id,
               a.journal AS journal,
               a.title AS title
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, doi=doi)
            return list(result)

    def query_by_title(self, title_search):
        """Query by title (substring match)."""
        query = """
        MATCH (a:PubMedArticle)
        WHERE toLower(a.title) CONTAINS toLower($title_search)
          AND a.`local-url` IS NOT NULL
        RETURN a.pubmed_id AS pubmed_id,
               a.journal AS journal,
               a.title AS title
        LIMIT 10
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, title_search=title_search)
            return list(result)

    def determine_query_type(self, user_input):
        """Determine the type of query based on input format."""
        user_input = user_input.strip()

        # Check if numeric (PubMed ID)
        if user_input.isdigit():
            return 'pubmed_id', user_input

        # Check if DOI (starts with "10.")
        if user_input.startswith("10."):
            return 'doi', user_input

        # Otherwise, treat as title
        return 'title', user_input

    def search(self, user_input):
        """Execute search based on input type."""
        query_type, search_value = self.determine_query_type(user_input)

        if query_type == 'pubmed_id':
            print(f"Searching by PubMed ID: {search_value}")
            return self.query_by_pubmed_id(search_value)
        elif query_type == 'doi':
            print(f"Searching by DOI: {search_value}")
            return self.query_by_doi(search_value)
        else:
            print(f"Searching by title: '{search_value}'")
            return self.query_by_title(search_value)

    def display_results(self, results):
        """Display query results in a formatted manner."""
        if not results:
            print("\n✗ Paper NOT found in your ReadCube repository")
            print("  (You can safely download this paper)\n")
            return

        print(f"\n✓ Found {len(results)} paper(s) in your ReadCube repository:")
        print("=" * 80)

        for idx, record in enumerate(results, 1):
            print(f"\n[{idx}]")
            print(f"  PubMed ID: {record['pubmed_id']}")
            print(f"  Journal:   {record['journal'] or 'N/A'}")
            print(f"  Title:     {record['title']}")

        print("=" * 80)
        print()


def main():
    """Main execution function."""
    print("=" * 80)
    print("ReadCube Papers Repository Query Tool")
    print("=" * 80)
    print()

    # Get Neo4j credentials from environment variables
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_user = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')  # Default to 'neo4j'

    # Validate credentials
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        print("✗ Error: Missing required environment variables")
        print("  Please set: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
        print("  Optional: NEO4J_DATABASE (defaults to 'neo4j')")
        sys.exit(1)

    # Initialize query tool
    query_tool = ReadCubeQueryTool(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)

    try:
        print("Enter one of the following to search:")
        print("  • PubMed ID (numeric, e.g., 19308067)")
        print("  • DOI (starts with '10.', e.g., 10.1038/nrc2622)")
        print("  • Title (full or partial, e.g., 'Metastasis')")
        print("\nType 'STOP' to exit\n")

        # Main query loop
        while True:
            user_input = input("Search> ").strip()

            if user_input.upper() == 'STOP':
                print("\nExiting ReadCube Query Tool. Goodbye!")
                break

            if not user_input:
                print("Please enter a search term or 'STOP' to exit\n")
                continue

            try:
                results = query_tool.search(user_input)
                query_tool.display_results(results)
            except Exception as e:
                print(f"\n✗ Error executing query: {e}\n")

    finally:
        query_tool.close()


if __name__ == "__main__":
    main()