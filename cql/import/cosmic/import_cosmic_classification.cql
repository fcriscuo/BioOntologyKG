// COSMIC Classification Data Import Script for Neo4j v5
// This script imports COSMIC classification data, creates nodes,
// and establishes relationships with existing NCIT_Term nodes.
// n.b. The classification data is keyed on PHENOTYPE_ID
//       These data should not be imported before the NCIT_Term data

// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create a unique constraint on the COSMIC_PHENOTYPE_ID property.
CREATE CONSTRAINT cosmic_phenotype_id_unique IF NOT EXISTS
FOR (c:COSMIC_Classification) REQUIRE c.COSMIC_PHENOTYPE_ID IS UNIQUE;

// ================================
// STEP 2: Clean Existing Data
// ================================
// Detach and delete all existing COSMIC_Classification nodes in batches.
MATCH (c:COSMIC_Classification)
CALL (c) {
    DETACH DELETE c
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 3: Import Data and Create Relationships
// ================================
// Load data from the TSV file, create COSMIC_Classification nodes,
// and link them to NCIT_Term nodes.
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/COSMIC_rel102/Cosmic_Classification_Tsv_v102_GRCh38/Cosmic_Classification_v102_GRCh38.tsv' AS row FIELDTERMINATOR '\t'
CALL (row){
    // Create the COSMIC_Classification node and set its properties
    CREATE (c:COSMIC_Classification)
    SET c = row

    // If an NCI_CODE exists, match the NCIT_Term and create the relationship
    WITH c, row
    WHERE row.NCI_CODE IS NOT NULL AND row.NCI_CODE <> ''
    MATCH (n:NCIT_Term {NCI_CODE: row.NCI_CODE})
    MERGE (c)-[:HAS_NCIT]->(n)

} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 4: Verification Queries
// ================================
// Count the total number of COSMIC_Classification nodes imported.
MATCH (c:COSMIC_Classification)
RETURN count(c) AS totalNodes;

// Count the total number of HAS_NCIT relationships created.
MATCH ()-[r:HAS_NCIT]->()
RETURN count(r) AS totalRelationships;

// Sample a few nodes and their relationships to verify the import.
MATCH (c:COSMIC_Classification)-[:HAS_NCIT]->(n:NCIT_Term)
RETURN c.COSMIC_PHENOTYPE_ID, n.NCI_CODE, n.term
LIMIT 10;

// ================================
// SCRIPT COMPLETION MESSAGE
// ================================
RETURN "COSMIC Classification import script completed successfully. Verify results using the queries above." AS status;
