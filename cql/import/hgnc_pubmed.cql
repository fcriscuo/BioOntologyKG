
// HGNC to Publication Relationship Script for Neo4j v5
// This script creates HAS_PUBLICATION relationships between existing HGNC nodes
// and new or existing Publication nodes based on PubMed identifiers.

// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create a unique constraint on pubmed_id for Publication nodes to ensure data integrity
// and optimize relationship creation.
CREATE CONSTRAINT publication_pubmed_id_unique IF NOT EXISTS
FOR (p:Publication) REQUIRE p.pubmed_id IS UNIQUE;

CREATE CONSTRAINT publication_unique IF NOT EXISTS
FOR ()-[r:HAS_PUBLICATION]-()
REQUIRE (r.source, r.target) IS UNIQUE;

// ================================
// STEP 2: Create Relationships
// ================================
// Process each HGNC node that has a pubmed_id property.
// Unwind the list of pubmed_ids to process each one individually.
// Use MERGE to create Publication nodes if they don't already exist.
// Use MERGE to create the HAS_PUBLICATION relationship, avoiding duplicates.
// This operation is batched for better performance and memory management.
MATCH (h:HGNC)
WHERE h.pubmed_id IS NOT NULL AND size(h.pubmed_id) > 0
CALL {
    WITH h
    UNWIND h.pubmed_id AS pubmedId
    MERGE (p:Publication {pubmed_id: pubmedId})
    MERGE (h)-[:HAS_PUBLICATION]->(p)
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 3: Verification Queries
// ================================
// Count the total number of HAS_PUBLICATION relationships created.
MATCH ()-[r:HAS_PUBLICATION]->()
RETURN count(r) AS totalRelationships;

// Sample a few HGNC nodes and their related publications to verify the import.
MATCH (h:HGNC)-[:HAS_PUBLICATION]->(p:Publication)
RETURN h.hgnc_id, collect(p.pubmed_id) AS publications
LIMIT 10;

// Verify that an HGNC node can have multiple publications.
MATCH (h:HGNC)
WHERE size((h)-[:HAS_PUBLICATION]->()) > 1
RETURN h.hgnc_id, COUNT((h)-[:HAS_PUBLICATION]->()) AS publicationCount
ORDER BY publicationCount DESC
LIMIT 5;

// Verify that a Publication can be linked to multiple HGNC nodes.
MATCH (p:Publication)
WHERE size((p)<-[:HAS_PUBLICATION]-()) > 1
RETURN p.pubmed_id, size((p)<-[:HAS_PUBLICATION]-()) AS hgncCount
ORDER BY hgncCount DESC
LIMIT 5;

// ================================
// SCRIPT COMPLETION MESSAGE
// ================================
RETURN "HGNC to Publication relationship script completed successfully. Verify results using the queries above." AS status;
