// ================================
// Import Disease Ontology (DOID) from OBO file
CALL n10s.graphconfig.init();
// Create uniqueness constraint on DOID terms
CREATE CONSTRAINT doid_term_id_unique IF NOT EXISTS
FOR (d:DOID_Term) REQUIRE d.id IS UNIQUE;
// Import DOID ontology using n10s plugin
// Ensure n10s plugin is installed and configured in Neo4j
// The import fetch command retrieves and imports the ontology from the specified URL
CALL n10s.onto.import.fetch("http://purl.obolibrary.org/obo/doid.owl","RDF/XML");

MATCH (d:Resource) WHERE d.uri STARTS WITH "http://purl.obolibrary.org/obo/doid"
SET d:DOID_Term,
d.id = split(d.uri, "/")[-1]
RETURN count(d) AS totalDOIDTerms;