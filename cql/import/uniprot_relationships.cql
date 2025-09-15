
// UniProt to Gene Ontology Relationship Script for Neo4j v5
// This script creates relationships between existing UniProt nodes and GoTerm nodes.

// ================================
// Create Relationships
// ================================
// UniProt -> GOTerm
// Process each UniProt node, unwind its Gene Ontology IDs, and create a dynamic relationship
// to the corresponding GoTerm node. The relationship type is based on the GoTerm's namespace.
MATCH (u:UniProt)
WHERE u.`Gene_Ontology_IDs` IS NOT NULL AND size(u.`Gene_Ontology_IDs`) > 0
WITH u
UNWIND u.Gene_Ontology_IDs as goId
MATCH (g:GOTerm {id:trim(goId)})
CALL apoc.create.relationship(u, ("HAS_" + upper(g.namespace)), {}, g) YIELD rel
RETURN COUNT(g)

// HGNC -> UniProt
// Use the HGNC ID from UniProt nodes to link to existing HGNC nodes.
MATCH (u:UniProt)
WHERE (u.HGNC) IS NOT NULL
WITH u
MATCH (h:HGNC{hgnc_id:u.HGNC[0]})
CALL apoc.create.relationship(h,"HAS_UNIPROT",{},u) YIELD rel
RETURN COUNT(h)

// HGNC -> GoTerm
// Have to use the related UniProt node as an intermediary
MATCH (h:HGNC)
WHERE h.uniprots_id IS NOT NULL
WITH h
MATCH (u:UniProt) WHERE u.Entry IN h.uniprots_id
UNWIND u.Gene_Ontology_IDs as goId
MATCH (g:GOTerm {id:trim(goId)})
CALL apoc.create.relationship(h, ("HAS_" + upper(g.namespace)), {}, g) YIELD rel
RETURN COUNT(g)



// UniProt -> PubMed
MATCH (u:UniProt)
WHERE u.PubMed_ID IS NOT NULL AND size(u.PubMed_ID) > 0
CALL (u) {
    WITH u
     UNWIND u.PubMed_ID AS pubmedId
     MERGE (p:Publication {pubmed_id: pubmedId})
     MERGE (u)-[:HAS_PUBLICATION]->(p)
} IN TRANSACTIONS OF 1000 ROWS;
// ================================
// STEP 2: Verification Queries
// ================================
// Count the total number of relationships created between UniProt and GoTerm nodes.
MATCH (:UniProt)-[r]->(:GoTerm)
WHERE type(r) STARTS WITH 'HAS_'
RETURN count(r) AS totalRelationships;

// Sample a few UniProt nodes and their related GoTerms to verify the import.
MATCH (u:UniProt)-[r]->(g:GoTerm)
WHERE type(r) STARTS WITH 'HAS_'
RETURN u.Entry, type(r) AS relationshipType, g.id AS goTermId, g.name AS goTermName
LIMIT 10;

// Verify the dynamic relationship types.
MATCH (:UniProt)-[r]->(:GoTerm)
WHERE type(r) STARTS WITH 'HAS_'
RETURN DISTINCT type(r) AS relationshipType, count(r) AS count
ORDER BY count DESC;

// ================================
// SCRIPT COMPLETION MESSAGE
// ================================
RETURN "UniProt to GoTerm relationship script completed successfully. Verify results using the queries above." AS status;
