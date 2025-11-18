"""
OBO to Neo4j Importer
Loads OBO-formatted ontology files into Neo4j database

Requirements:
    pip install pronto neo4j obonet networkx
"""
import os
import pronto
from neo4j import GraphDatabase
import obonet
import networkx as nx
from typing import Optional


class OBONeo4jImporter:
    """Import OBO ontology data into Neo4j database"""

    def __init__(self, uri: str, user: str, password: str, ontology_label: Optional[str] = None):
        """
        Initialize Neo4j connection

        Args:
            uri: Neo4j connection URI (e.g., 'bolt://localhost:7687')
            user: Neo4j username
            password: Neo4j password
            ontology_label: Additional label for nodes (e.g., 'SequenceOntologyTerm')
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.ontology_label = ontology_label

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()

    def clear_sequence_ontology(self):
        """Clear all SequenceOntologyTerm nodes and edges"""
        with self.driver.session() as session:
           session.run("MATCH (so:SequenceOntologyTerm) DETACH DELETE so")
           print("SequenceOntologyTerm nodes and relationships cleared")



    def import_with_pronto(self, obo_file_path: str, batch_size: int = 1000):
        """
        Import OBO file using pronto library

        Args:
            obo_file_path: Path to OBO file
            batch_size: Number of nodes to create per batch
        """
        print(f"Loading OBO file with pronto: {obo_file_path}")
        ontology = pronto.Ontology(obo_file_path)

        with self.driver.session() as session:
            # Create constraint for unique term IDs
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Term) REQUIRE t.id IS UNIQUE")

            term_batch = []
            rel_batch = []

            print(f"Processing {len(ontology)} terms...")

            for i, term in enumerate(ontology.terms()):
                # Prepare term properties
                term_props = {
                    'id': term.id,
                    'name': term.name or '',
                    'definition': term.definition or '',
                    'namespace': term.namespace or '',
                    'is_obsolete': term.obsolete
                }

                # Add synonyms if present
                if term.synonyms:
                    term_props['synonyms'] = [syn.description for syn in term.synonyms]

                # Add cross-references if present
                if term.xrefs:
                    term_props['xrefs'] = [str(xref) for xref in term.xrefs]

                term_batch.append(term_props)

                # Process relationships
                for parent in term.superclasses(distance=1, with_self=False):
                    rel_batch.append({
                        'from_id': term.id,
                        'to_id': parent.id,
                        'type': 'IS_A'
                    })

                # Process other relationships
                for rel_type, targets in term.relationships.items():
                    for target in targets:
                        rel_batch.append({
                            'from_id': term.id,
                            'to_id': target.id,
                            'type': str(rel_type).upper().replace(':', '_')
                        })

                # Batch insert
                if len(term_batch) >= batch_size:
                    self._create_terms(session, term_batch)
                    term_batch = []

                if len(rel_batch) >= batch_size:
                    self._create_relationships(session, rel_batch)
                    rel_batch = []

                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1} terms...")

            # Insert remaining batches
            if term_batch:
                self._create_terms(session, term_batch)
            if rel_batch:
                self._create_relationships(session, rel_batch)

            print("Import completed!")

    def import_with_obonet(self, obo_file_path: str):
        """
        Import OBO file using obonet library

        Args:
            obo_file_path: Path to OBO file
        """
        print(f"Loading OBO file with obonet: {obo_file_path}")
        graph = obonet.read_obo(obo_file_path)

        with self.driver.session() as session:
            # Create constraint
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Term) REQUIRE t.id IS UNIQUE")

            print(f"Processing {graph.number_of_nodes()} terms...")

            term_batch = []
            for node_id, data in graph.nodes(data=True):
                term_props = {
                    'id': node_id,
                    'name': data.get('name', ''),
                }

                # Add optional properties
                if 'def' in data:
                    term_props['definition'] = data['def']
                if 'namespace' in data:
                    term_props['namespace'] = data['namespace']
                if 'is_obsolete' in data:
                    term_props['is_obsolete'] = data['is_obsolete']
                if 'synonym' in data:
                    term_props['synonyms'] = data['synonym']
                if 'xref' in data:
                    term_props['xrefs'] = data['xref']

                term_batch.append(term_props)

            self._create_terms(session, term_batch)

            print("Creating relationships...")
            rel_batch = []
            for source, target, data in graph.edges(data=True):
                rel_batch.append({
                    'from_id': source,
                    'to_id': target,
                    'type': data.get('type', 'IS_A').upper().replace(':', '_')
                })

            self._create_relationships(session, rel_batch)
            print("Import completed!")

    def _create_terms(self, session, term_batch):
        """Create term nodes in batch"""
        # Build labels string
        labels = "Term"
        if self.ontology_label:
            labels += f":{self.ontology_label}"

        query = f"""
        UNWIND $batch AS term
        MERGE (t:{labels} {{id: term.id}})
        SET t += term
        """
        session.run(query, batch=term_batch)
        print(f"Created {len(term_batch)} terms")

    def _create_relationships(self, session, rel_batch):
        """Create relationships in batch"""
        query = """
        UNWIND $batch AS rel
        MATCH (from:Term {id: rel.from_id})
        MATCH (to:Term {id: rel.to_id})
        CALL apoc.create.relationship(from, rel.type, {}, to) YIELD rel as r
        RETURN count(r)
        """

        # Fallback if APOC is not available
        try:
            session.run(query, batch=rel_batch)
        except:
            # Create IS_A relationships directly (most common)
            is_a_batch = [r for r in rel_batch if r['type'] == 'IS_A']
            if is_a_batch:
                query_is_a = """
                UNWIND $batch AS rel
                MATCH (from:Term {id: rel.from_id})
                MATCH (to:Term {id: rel.to_id})
                MERGE (from)-[:IS_A]->(to)
                """
                session.run(query_is_a, batch=is_a_batch)

            # Handle other relationship types
            other_rels = [r for r in rel_batch if r['type'] != 'IS_A']
            for rel in other_rels:
                query_dynamic = f"""
                MATCH (from:Term {{id: $from_id}})
                MATCH (to:Term {{id: $to_id}})
                MERGE (from)-[:{rel['type']}]->(to)
                """
                session.run(query_dynamic, from_id=rel['from_id'], to_id=rel['to_id'])

        print(f"Created {len(rel_batch)} relationships")

    def get_statistics(self):
        """Get database statistics"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Term)
                WITH count(t) as term_count
                MATCH ()-[r]->()
                RETURN term_count, count(r) as rel_count
            """)
            record = result.single()
            if record:
                print(f"\nDatabase Statistics:")
                print(f"  Terms: {record['term_count']}")
                print(f"  Relationships: {record['rel_count']}")


# Example usage
if __name__ == "__main__":
    # Configuration
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USER = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')  # Replace with your password

    OBO_FILE = "/Volumes/SSD870/data/SequenceOntology/so.obo"

    # Initialize importer with optional ontology-specific label
    # For Sequence Ontology:
    importer = OBONeo4jImporter(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        ontology_label="SequenceOntologyTerm"  # Adds second label to all nodes
    )

    # For Gene Ontology, you might use:
    # importer = OBONeo4jImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    #                              ontology_label="GeneOntologyTerm")

    # Or without additional label (just :Term):
    # importer = OBONeo4jImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # Optional: Clear existing data
        importer.clear_sequence_ontology()

        # Import using pronto (recommended for complex ontologies)
        #importer.import_with_pronto(OBO_FILE)

        # OR import using obonet (faster for simpler ontologies)
        importer.import_with_obonet(OBO_FILE)

        # Show statistics
        importer.get_statistics()

    finally:
        importer.close()