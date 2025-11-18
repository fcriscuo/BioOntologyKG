#!/usr/bin/env python3
"""
MonarchKG Relationship Data Importer for Neo4j
Imports TSV relationship files filtered by human taxon (NCBITaxon:9606)
This Python script was generated using Clause Sonnet 4.5
The prompt is documented in prompts/import/monarchkg/monarch_relationships_generic_prompt.txt
"""

import os
import csv
from neo4j import GraphDatabase
from typing import List, Dict, Optional


class MonarchKGImporter:
    def __init__(self, uri: str, user: str, password: str, database: str):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.relationships_created = 0

    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()

    def parse_publications(self, pub_string: str) -> Optional[List[str]]:
        """Parse publications string into a list"""
        if not pub_string or pub_string.strip() == '':
            return None

        # Remove brackets and quotes, split by comma
        pub_string = pub_string.strip()
        if pub_string.startswith('[') and pub_string.endswith(']'):
            pub_string = pub_string[1:-1]

        # Split and clean each publication
        pubs = [p.strip().strip('"\'') for p in pub_string.split(',')]
        pubs = [p for p in pubs if p]  # Remove empty strings

        return pubs if pubs else None

    def create_relationship_batch(self, tx, batch: List[Dict]):
        """Create a batch of relationships in a single transaction"""
        query = """
        UNWIND $batch AS row
        MERGE (s {id: row.subject_id})
        ON CREATE SET s:`${subject_label}`
        MERGE (t {id: row.object_id})
        ON CREATE SET t:`${object_label}`
        CREATE (s)-[r:`${predicate}`]->(t)
        SET r.primary_knowledge_source = row.primary_knowledge_source
        """

        # Process each row in the batch
        for row_data in batch:
            # Build the query dynamically for each relationship
            cypher = f"""
            MERGE (s:`{row_data['subject_category']}` {{id: $subject_id}})
            MERGE (t:`{row_data['object_category']}` {{id: $object_id}})
            CREATE (s)-[r:`{row_data['predicate']}`]->(t)
            SET r.primary_knowledge_source = $primary_knowledge_source
            """

            # Add publications if they exist
            if row_data.get('publications'):
                cypher += "\nSET r.publications = $publications"

            params = {
                'subject_id': row_data['subject_id'],
                'object_id': row_data['object_id'],
                'primary_knowledge_source': row_data['primary_knowledge_source']
            }

            if row_data.get('publications'):
                params['publications'] = row_data['publications']

            tx.run(cypher, **params)

        return len(batch)

    def import_relationships(self, file_path: str, batch_size: int = 1000):
        """Import relationships from TSV file"""
        print(f"Starting import from: {file_path}")
        print(f"Filtering for subject_taxon: NCBITaxon:9606")
        print(f"Batch size: {batch_size}")
        print("-" * 60)

        batch = []
        total_rows = 0
        filtered_rows = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')

                for row in reader:
                    total_rows += 1

                    # Filter by subject_taxon
                    if row.get('subject_taxon') != 'NCBITaxon:9606':
                        continue

                    filtered_rows += 1

                    # Parse publications
                    publications = self.parse_publications(row.get('publications', ''))

                    # Prepare row data
                    row_data = {
                        'subject_id': row['subject'],
                        'subject_category': row['subject_category'],
                        'predicate': row['predicate'],
                        'object_id': row['object'],
                        'object_category': row['object_category'],
                        'primary_knowledge_source': row.get('primary_knowledge_source', ''),
                        'publications': publications
                    }

                    batch.append(row_data)

                    # Process batch when it reaches batch_size
                    if len(batch) >= batch_size:
                        with self.driver.session(database=self.database) as session:
                            created = session.execute_write(
                                self.create_relationship_batch, batch
                            )
                            self.relationships_created += created

                        print(f"Processed {filtered_rows} rows (Total scanned: {total_rows}), "
                              f"Created {self.relationships_created} relationships")
                        batch = []

                # Process remaining batch
                if batch:
                    with self.driver.session(database=self.database) as session:
                        created = session.execute_write(
                            self.create_relationship_batch, batch
                        )
                        self.relationships_created += created
                    batch = []

        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return False
        except Exception as e:
            print(f"Error during import: {str(e)}")
            return False

        print("-" * 60)
        print(f"Import completed!")
        print(f"Total rows scanned: {total_rows}")
        print(f"Rows matching filter: {filtered_rows}")
        print(f"Relationships created: {self.relationships_created}")

        return True


def get_neo4j_config() -> Dict[str, str]:
    """Get Neo4j connection parameters from environment variables"""
    config = {
        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'user': os.getenv('NEO4J_USER', 'neo4j'),
        'password': os.getenv('NEO4J_PASSWORD'),
        'database': os.getenv('NEO4J_DATABASE', 'neo4j')
    }

    if not config['password']:
        raise ValueError("NEO4J_PASSWORD environment variable must be set")

    return config


def main():
    """Main execution function"""
    print("=" * 60)
    print("MonarchKG Relationship Importer for Neo4j")
    print("=" * 60)
    print()

    # Get file path from user
    file_path = input("Enter the full path to the MonarchKG TSV file: ").strip()

    if not file_path:
        print("Error: No file path provided")
        return

    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}")
        return

    print()

    # Get Neo4j configuration
    try:
        config = get_neo4j_config()
        print(f"Neo4j URI: {config['uri']}")
        print(f"Neo4j User: {config['user']}")
        print(f"Neo4j Database: {config['database']}")
        print()
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
        print("\nPlease set the following environment variables:")
        print("  NEO4J_URI (default: bolt://localhost:7687)")
        print("  NEO4J_USER (default: neo4j)")
        print("  NEO4J_PASSWORD (required)")
        print("  NEO4J_DATABASE (default: neo4j)")
        return

    # Create importer and run import
    importer = None
    try:
        importer = MonarchKGImporter(
            uri=config['uri'],
            user=config['user'],
            password=config['password'],
            database=config['database']
        )

        success = importer.import_relationships(file_path)

        if not success:
            print("\nImport failed!")
            return

    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        return
    finally:
        if importer:
            importer.close()
            print("\nNeo4j connection closed.")


if __name__ == "__main__":
    main()