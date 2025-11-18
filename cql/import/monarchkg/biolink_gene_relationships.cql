// ============================================================================
// Neo4j Cypher Script: Create Relationships for Biolink Gene Nodes
// Description: Creates HAS_SEQUENCE_ONTOLOGY, HAS_HGNC, and HAS_BIOLINK_GENE
//              relationships with batched transactions
// ============================================================================

// CALL Block 1: Create relationships FROM biolink:Gene nodes
// This block processes HAS_SEQUENCE_ONTOLOGY and HAS_HGNC relationships

  MATCH (bg:`biolink:Gene` {category: 'biolink:Gene'})
  //WITH bg
  CALL (bg) {
    WITH bg
    // Create HAS_SEQUENCE_ONTOLOGY relationship
    MATCH (so:SequenceOntologyTerm)
    WHERE bg.type = so.id
    MERGE (bg)-[:HAS_SEQUENCE_ONTOLOGY]->(so)
  } IN TRANSACTIONS OF 1000 ROWS

MATCH (bg:`biolink:Gene` {category: 'biolink:Gene'})
  CALL (bg) {
    WITH bg
    // Create HAS_HGNC relationship
    MATCH (hgnc:HGNC)
    WHERE bg.id = hgnc.hgnc_id
    MERGE (bg)-[:HAS_HGNC]->(hgnc)
  } IN TRANSACTIONS OF 1000 ROWS

// CALL Block 2: Create HAS_BIOLINK_GENE relationships FROM Gene nodes
// This block connects legacy Gene nodes to biolink:Gene nodes
 MATCH (g:Gene)
CALL (g) {
  //WITH g
  //CALL {
   // WITH g
    MATCH (bg:`biolink:Gene` {category: 'biolink:Gene'})
    WHERE g.GENE_SYMBOL = bg.symbol
    MERGE (g)-[:HAS_BIOLINK_GENE]->(bg)
  } IN TRANSACTIONS OF 1000 ROWS

  RETURN count(g) as legacyGenesProcessed
}
RETURN legacyGenesProcessed;

// ============================================================================
// Expected Output:
// - biolinkGenesProcessed: Count of biolink:Gene nodes processed
// - legacyGenesProcessed: Count of legacy Gene nodes processed
// ============================================================================