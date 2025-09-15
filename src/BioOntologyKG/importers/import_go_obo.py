#!/usr/bin/env python3
"""
Gene Ontology Parser using Pronto for Neo4j Import
Parses GO OBO file and extracts terms and relationships for Neo4j database
The script imports data directl from the GeneOntology go.obo file
This code was generated Claude with some minor revisions
"""
import os

import pronto
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GOTerm:
    """Data class representing a Gene Ontology term"""
    id: str
    name: str
    namespace: str
    definition: str = ""
    is_obsolete: bool = False
    synonyms: List[str] = None
    xrefs: List[str] = None
    alt_ids: List[str] = None

    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []
        if self.alt_ids is None:
            self.alt_ids = []


@dataclass
class GORelationship:
    """Data class representing a relationship between GO terms"""
    subject_id: str
    predicate: str
    object_id: str


class GOParser:
    """Parser for Gene Ontology OBO files using Pronto"""

    def __init__(self, obo_file_path: str):
        """
        Initialize parser with OBO file path

        Args:
            obo_file_path: Path to the GO OBO file (local file or URL)
        """
        self.obo_file_path = obo_file_path
        self.ontology = None
        self.terms: Dict[str, GOTerm] = {}
        self.relationships: List[GORelationship] = []

    def load_ontology(self):
        """Load the ontology from OBO file"""
        logger.info(f"Loading ontology from: {self.obo_file_path}")
        try:
            # Pronto can handle both local files and URLs
            self.ontology = pronto.Ontology(self.obo_file_path)
            logger.info(f"Successfully loaded ontology with {len(self.ontology)} terms")
        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")
            raise

    def parse_terms(self):
        """Extract all GO terms from the ontology"""
        logger.info("Parsing GO terms...")

        for term in self.ontology.terms():
            # Skip root terms or terms without proper IDs
            if not term.id or term.id.startswith('http'):
                continue

            # Extract synonyms
            synonyms = []
            if hasattr(term, 'synonyms') and term.synonyms:
                synonyms = [str(syn) for syn in term.synonyms]

            # Extract xrefs
            xrefs = []
            if hasattr(term, 'xrefs') and term.xrefs:
                xrefs = [str(xref) for xref in term.xrefs]

            # Extract alternative IDs
            alt_ids = []
            if hasattr(term, 'alternate_ids') and term.alternate_ids:
                alt_ids = list(term.alternate_ids)

            # Create GOTerm object
            go_term = GOTerm(
                id=term.id,
                name=term.name if term.name else "",
                namespace=getattr(term, 'namespace', ''),
                definition=term.definition if term.definition else "",
                is_obsolete=getattr(term, 'obsolete', False),
                synonyms=synonyms,
                xrefs = xrefs,
                alt_ids=alt_ids
            )

            self.terms[term.id] = go_term

        logger.info(f"Parsed {len(self.terms)} terms")

    def parse_relationships(self):
        """Extract relationships between GO terms"""
        logger.info("Parsing relationships...")

        relationship_count = 0

        for term in self.ontology.terms():
            if not term.id or term.id.startswith('http'):
                continue

            # Parse 'is_a' relationships (subclass relationships)
            for parent in term.superclasses(distance=1):
                if parent.id and not parent.id.startswith('http'):
                    self.relationships.append(
                        GORelationship(term.id, "is_a", parent.id)
                    )
                    relationship_count += 1

            # Parse other relationships (part_of, regulates, etc.)
            if hasattr(term, 'relationships'):
                for rel_type, related_terms in term.relationships.items():
                    rel_name = str(rel_type).split('/')[-1]  # Extract relation name

                    for related_term in related_terms:
                        if related_term.id and not related_term.id.startswith('http'):
                            self.relationships.append(
                                GORelationship(term.id, rel_name, related_term.id)
                            )
                            relationship_count += 1

        logger.info(f"Parsed {relationship_count} relationships")

    def get_terms_by_namespace(self, namespace: str) -> Dict[str, GOTerm]:
        """Get terms filtered by namespace (biological_process, molecular_function, cellular_component)"""
        return {
            term_id: term for term_id, term in self.terms.items()
            if term.namespace == namespace
        }

    def get_term_statistics(self) -> Dict[str, int]:
        """Get statistics about the parsed terms"""
        stats = {
            'total_terms': len(self.terms),
            'total_relationships': len(self.relationships),
            'biological_process': len(self.get_terms_by_namespace('biological_process')),
            'molecular_function': len(self.get_terms_by_namespace('molecular_function')),
            'cellular_component': len(self.get_terms_by_namespace('cellular_component')),
            'obsolete_terms': sum(1 for term in self.terms.values() if term.is_obsolete)
        }
        return stats


class Neo4jImporter:
    """Imports parsed GO data into Neo4j database"""

    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize Neo4j connection

        Args:
            uri: Neo4j database URI (e.g., "bolt://localhost:7687")
            username: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        """Close the database connection"""
        self.driver.close()

    def create_constraints(self):
        """Create constraints and indexes for better performance"""
        with self.driver.session() as session:
            # Create constraint on GO term ID
            session.run("""
                        CREATE CONSTRAINT go_term_id IF NOT EXISTS
                        FOR (t:GOTerm) REQUIRE t.id IS UNIQUE
                        """)

            # Create index on term name
            session.run("""
                CREATE INDEX go_term_name IF NOT EXISTS
                FOR (t:GOTerm) ON (t.name)
            """)

            # Create index on namespace
            session.run("""
                CREATE INDEX go_term_namespace IF NOT EXISTS
                FOR (t:GOTerm) ON (t.namespace)
            """)

        logger.info("Created constraints and indexes")

    def import_terms(self, terms: Dict[str, GOTerm], batch_size: int = 1000):
        """Import GO terms into Neo4j"""
        logger.info(f"Importing {len(terms)} terms...")

        term_list = list(terms.values())

        with self.driver.session() as session:
            for i in range(0, len(term_list), batch_size):
                batch = term_list[i:i + batch_size]

                session.run("""
                            UNWIND $terms AS term
                            MERGE (t:GOTerm {id: term.id})
                            SET t.name = term.name,
                            t.namespace = term.namespace,
                            t.definition = term.definition,
                            t.is_obsolete = term.is_obsolete,
                            t.synonyms = term.synonyms,
                            t.xrefs = term.xrefs,
                            t.alt_ids = term.alt_ids
                            """, terms=[{
                    'id': term.id,
                    'name': term.name,
                    'namespace': term.namespace,
                    'definition': term.definition,
                    'is_obsolete': term.is_obsolete,
                    'synonyms': term.synonyms,
                    'alt_ids': term.alt_ids
                } for term in batch])

                logger.info(f"Imported batch {i // batch_size + 1}/{(len(term_list) - 1) // batch_size + 1}")

    def import_relationships(self, relationships: List[GORelationship], batch_size: int = 1000):
        """Import relationships between GO terms"""
        logger.info(f"Importing {len(relationships)} relationships...")

        with self.driver.session() as session:
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]

                session.run("""
                            UNWIND $rels AS rel
                            MATCH (subject:GOTerm {id: rel.subject_id})
                            MATCH (object:GOTerm {id: rel.object_id})
                            CALL apoc.create.relationship(subject, rel.predicate, {}, object) YIELD rel AS r
                            RETURN count(r)
                            """, rels=[{
                    'subject_id': rel.subject_id,
                    'predicate': rel.predicate.upper(),
                    'object_id': rel.object_id
                } for rel in batch])

                logger.info(
                    f"Imported relationship batch {i // batch_size + 1}/{(len(relationships) - 1) // batch_size + 1}")


def main():
    """Main function demonstrating the complete workflow"""

    # Configuration
    OBO_FILE_PATH = "http://purl.obolibrary.org/obo/go.obo"  # or local file path
    neo4j_uri = os.getenv("neo4j_uri")
    neo4j_username = os.getenv("neo4j_username")
    neo4j_password = os.getenv("neo4j_password")

    try:
        # Parse GO data
        parser = GOParser(OBO_FILE_PATH)
        parser.load_ontology()
        parser.parse_terms()
        parser.parse_relationships()

        # Print statistics
        stats = parser.get_term_statistics()
        logger.info("Parsing Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # Import to Neo4j (uncomment to use)

        importer = Neo4jImporter(neo4j_uri, neo4j_username, neo4j_password)
        importer.create_constraints()
        importer.import_terms(parser.terms)
        importer.import_relationships(parser.relationships)
        importer.close()


        # Example: Get all biological process terms
        bp_terms = parser.get_terms_by_namespace('biological_process')
        logger.info(f"Found {len(bp_terms)} biological process terms")

        # Example: Show first few terms
        for i, (term_id, term) in enumerate(list(parser.terms.items())[:5]):
            print(f"\nTerm {i + 1}:")
            print(f"  ID: {term.id}")
            print(f"  Name: {term.name}")
            print(f"  Namespace: {term.namespace}")
            print(f"  Definition: {term.definition[:100]}..." if len(
                term.definition) > 100 else f"  Definition: {term.definition}")
            print(f"  Synonyms: {term.synonyms[:3]}...")  # Show first 3 synonyms
            print(f"  Xrefs: {term.xrefs[:3]}...")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()