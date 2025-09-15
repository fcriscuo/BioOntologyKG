// COSMIC Genes Data Import Script for Neo4j v5
// This script imports gene data, creates nodes, and establishes relationships
// to HGNC, COSMIC_Cancer_Gene, and UniProt nodes.

// ================================
// STEP 1: Create Unique Constraints
// ================================
CREATE CONSTRAINT cosmic_gene_id_unique IF NOT EXISTS
FOR (g:COSMIC_Gene) REQUIRE g.COSMIC_GENE_ID IS UNIQUE;

CREATE CONSTRAINT cosmic_gene_symbol_unique IF NOT EXISTS
FOR (g:COSMIC_Gene) REQUIRE g.GENE_SYMBOL IS UNIQUE;

// ================================
// STEP 2: Clean Existing Data
// ================================
// Detach and delete all existing COSMIC_Gene nodes in batches.
MATCH (g:COSMIC_Gene)
CALL (g) {
    DETACH DELETE g
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 3: Import Gene Nodes
// ================================
//:auto USING PERIODIC COMMIT 1000
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_Genes_v102_GRCh38.tsv' AS row FIELDTERMINATOR '\t'
CALL (row) {
// Create the COSMIC_Gene node
CREATE (g:COSMIC_Gene)
SET g = row
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 4: Create Relationships
// ================================

// Create HAS_HGNC relationships
MATCH (g:COSMIC_Gene)
WHERE g.HGNC_ID IS NOT NULL AND g.HGNC_ID <> ''
CALL (g){
    MATCH (h:HGNC {hgnc_id: 'HGNC:' + g.HGNC_ID})
    MERGE (g)-[:HAS_HGNC]->(h)
} IN TRANSACTIONS OF 1000 ROWS;

// Create HAS_COSMIC_CANCER_GENE relationships
MATCH (g:COSMIC_Gene)
WHERE g.IN_CANCER_CENSUS = 'y'
CALL (g) {
    MATCH (ccg:COSMIC_Cancer_Gene {COSMIC_GENE_ID: g.COSMIC_GENE_ID})
    MERGE (g)-[:HAS_COSMIC_CANCER_GENE]->(ccg)
} IN TRANSACTIONS OF 1000 ROWS;

// Create HAS_UNIPROT relationships
MATCH (g:COSMIC_Gene)
CALL (g){
    MATCH (u:UniProt)
    WHERE u.Entry_Name STARTS WITH g.GENE_SYMBOL
    MERGE (g)-[:HAS_UNIPROT]->(u)
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 5: Verification Queries
// ================================
// Count total nodes imported
MATCH (g:COSMIC_Gene)
RETURN count(g) AS totalNodes;

// Count relationships
MATCH ()-[r:HAS_HGNC]->()
RETURN count(r) AS totalHGNCRelationships;

MATCH ()-[r:HAS_COSMIC_CANCER_GENE]->()
RETURN count(r) AS totalCancerGeneRelationships;

MATCH ()-[r:HAS_UNIPROT]->()
RETURN count(r) AS totalUniProtRelationships;

// ================================
// SCRIPT COMPLETION MESSAGE
// ================================
RETURN "COSMIC Genes import script completed successfully." AS status;
============
// SCRIPT COMPLETION MESSAGE
// ================================
RETURN "COSMIC Genes import script completed successfully." AS status;
