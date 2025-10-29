// COSMIC Cancer Gene Census Import Script for Neo4j v2025
// This script imports gene census data, creates nodes
// n.b. The following nodes must be imported before executing this script:
//          COSMIC_Gene
// This CQL was created by Antropic Claude Sonnet 4.5

// ================================
// STEP 1: Create Unique Constraints
// ================================
CREATE CONSTRAINT cosmic_gene_id_unique IF NOT EXISTS
FOR (g:COSMIC_Cancer_Gene) REQUIRE g.COSMIC_GENE_ID IS UNIQUE;

CREATE CONSTRAINT cosmic_gene_symbol_unique IF NOT EXISTS
FOR (g:COSMIC_Cancer_Gene) REQUIRE g.GENE_SYMBOL IS UNIQUE;

// ================================
// STEP 2: Clean Existing Data
// ================================
// Detach and delete all existing COSMIC_Cancer_Gene nodes in batches.
MATCH (g:COSMIC_Cancer_Gene)
CALL {
    WITH g
    DETACH DELETE g
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 3: Import Gene Census Data
// Change the file name and path as required
// ================================

LOAD CSV WITH HEADERS FROM
  'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_CancerGeneCensus_v102_GRCh38.tsv' AS row FIELDTERMINATOR '\t'
CALL (row) {
    // Create the COSMIC_Cancer_Gene node
    CREATE (g:COSMIC_Cancer_Gene)
    // Set properties, handling type conversions and list splitting
    SET g = row,
        g.GENOME_START = toInteger(row.GENOME_START),
        g.GENOME_STOP = toInteger(row.GENOME_STOP),
        g.TUMOUR_TYPES_SOMATIC = CASE WHEN row.TUMOUR_TYPES_SOMATIC IS NOT NULL THEN split(row.TUMOUR_TYPES_SOMATIC, ',') ELSE [] END,
        g.TUMOUR_TYPES_GERMLINE = CASE WHEN row.TUMOUR_TYPES_GERMLINE IS NOT NULL THEN split(row.TUMOUR_TYPES_GERMLINE, ',') ELSE [] END,
        g.MUTATION_TYPES = CASE WHEN row.MUTATION_TYPES IS NOT NULL THEN split(row.MUTATION_TYPES, ',') ELSE [] END,
        g.TRANSLOCATION_PARTNER = CASE WHEN row.TRANSLOCATION_PARTNER IS NOT NULL THEN split(row.TRANSLOCATION_PARTNER, ',') ELSE [] END
    // create a relationship to COSMIC_GENE node
// Create HAS_COSMIC_CANCER_GENE relationships
    WITH g
    MATCH (c:COSMIC_Gene) {COSMIC_GENE_ID: g.COSMIC_GENE_ID})
    MERGE (g)-[:HAS_COSMIC_CANCER_GENE]->(c)

} IN TRANSACTIONS OF 1000 ROWS;


} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 4: Create Translocation Partner Relationships
// ================================
// After import, process the TRANSLOCATION_PARTNER list for each gene.
MATCH (g1:COSMIC_Cancer_Gene)
WHERE g1.TRANSLOCATION_PARTNER IS NOT NULL AND size(g1.TRANSLOCATION_PARTNER) > 0
CALL {
    WITH g1
    UNWIND g1.TRANSLOCATION_PARTNER AS partnerSymbol
    MATCH (g2:COSMIC_Cancer_Gene {GENE_SYMBOL: partnerSymbol})
    // Ensure relationship is created only once by comparing internal IDs
    WHERE id(g1) < id(g2)
    MERGE (g1)-[:HAS_TRANSLOCATION_PARTNER]->(g2)
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 5: Verification Queries
// ================================
// Count total nodes imported
MATCH (g:COSMIC_Cancer_Gene)
RETURN count(g) AS totalNodes;

// Count HAS_HGNC relationships
MATCH ()-[r:HAS_HGNC]->()
RETURN count(r) AS totalHGNCRelationships;

// Count HAS_TRANSLOCATION_PARTNER relationships
MATCH ()-[r:HAS_TRANSLOCATION_PARTNER]->()
RETURN count(r) AS totalTranslocationRelationships;

// Sample a translocation relationship
MATCH (g1:COSMIC_Cancer_Gene)-[:HAS_TRANSLOCATION_PARTNER]->(g2:COSMIC_Cancer_Gene)
RETURN g1.GENE_SYMBOL AS gene1, g2.GENE_SYMBOL AS gene2
LIMIT 10;

// ================================
// SCRIPT COMPLETION MESSAGE
// ================================
RETURN "COSMIC Cancer Gene Census import script completed successfully." AS status;
