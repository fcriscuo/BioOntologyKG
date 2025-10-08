// COSMIC Tumour Data Import Script for Neo4j v5
// This script imports tumour data from the COSMIC sample file, creates nodes,
// and links them to existing COSMIC_Individual nodes.

// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create a unique constraint on the TUMOUR_ID property.
CREATE CONSTRAINT cosmic_tumour_id_unique IF NOT EXISTS
FOR (t:COSMIC_Tumour) REQUIRE t.TUMOUR_ID IS UNIQUE;

// ================================
// STEP 2: Clean Existing Data
// ================================
// Detach and delete all existing COSMIC_Tumour nodes in batches.
MATCH (t:COSMIC_Tumour)
CALL (t) {
    DETACH DELETE t
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 3: Import Tumour Data and Create Relationships
// ================================
//:auto USING PERIODIC COMMIT 1000
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_Sample_v102_GRCh38.tsv' AS row FIELDTERMINATOR '\t'

CALL (row) {
// Create the COSMIC_Tumour node and set its properties
CREATE (t:COSMIC_Tumour)
SET t.TUMOUR_ID = row.TUMOUR_ID,
    t.INDIVIDUAL_ID = row.INDIVIDUAL_ID,
    t.TUMOUR_SOURCE = row.TUMOUR_SOURCE,
    t.NORMAL_TISSUE_TESTED = row.NORMAL_TISSUE_TESTED,
    t.AGE_AT_TUMOUR_RECURRENCE = toInteger(row.AGE_AT_TUMOUR_RECURRENCE),
    t.THERAPY_RELATIONSHIP = row.THERAPY_RELATIONSHIP,
    t.MUTATION_ALLELE_SPECIFICATION = row.MUTATION_ALLELE_SPECIFICATION,
    t.DRUG_RESPONSE = row.DRUG_RESPONSE,
    t.GRADE = row.GRADE,
    t.STAGE = row.STAGE,
    t.CYTOGENETICS = row.CYTOGENETICS,
    t.METASTATIC_SITE = row.METASTATIC_SITE,
    t.GERMLINE_MUTATION = row.GERMLINE_MUTATION,
    t.THERAPY = row.THERAPY

// Link the new tumour to the corresponding individual
WITH t, row
MATCH (i:COSMIC_Individual {INDIVIDUAL_ID: row.INDIVIDUAL_ID})
MERGE (i)-[:HAS_TUMOUR]->(t);
} IN TRANSACTIONS OF 1000 ROWS;
// ================================
// STEP 4: Verification Queries
// ================================
// Count total tumour nodes imported
MATCH (t:COSMIC_Tumour)
RETURN count(t) AS totalTumours;

// Count total HAS_TUMOUR relationships
MATCH ()-[r:HAS_TUMOUR]->()
RETURN count(r) AS totalRelationships;

// Sample a few individuals and their tumours
MATCH (i:COSMIC_Individual)-[:HAS_TUMOUR]->(t:COSMIC_Tumour)
RETURN i.INDIVIDUAL_ID, collect(t.TUMOUR_ID) AS tumours
LIMIT 10;

// ================================
// SCRIPT COMPLETION MESSAGE
// ================================
RETURN "COSMIC Tumour import script completed successfully." AS status;
