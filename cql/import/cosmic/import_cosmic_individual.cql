// COSMIC Individual Data Import Script for Neo4j v5 (Regenerated)
// This script imports a subset of columns from the COSMIC sample data
// to create COSMIC_Individual nodes.

// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create a unique constraint on the INDIVIDUAL_ID property to ensure data integrity.
CREATE CONSTRAINT cosmic_individual_id_unique IF NOT EXISTS
FOR (i:COSMIC_Individual) REQUIRE i.INDIVIDUAL_ID IS UNIQUE;

// ================================
// STEP 2: Clean Existing Data
// ================================
// Detach and delete all existing COSMIC_Individual nodes to facilitate re-running the script.
MATCH (i:COSMIC_Individual)
CALL {
    WITH i
    DETACH DELETE i
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 3: Import Individual Data
// ================================
:auto USING PERIODIC COMMIT 1000
LOAD CSV WITH HEADERS FROM 'file:///COSMIC_rel102/Import/Cosmic_Sample_v102_GRCh38.tsv' AS row FIELDTERMINATOR '\t'

// Create the COSMIC_Individual node. The unique constraint from Step 1 ensures no duplicates are created.
CREATE (i:COSMIC_Individual)
// Set only the specified properties, with type conversion for age fields
SET i.INDIVIDUAL_ID = row.INDIVIDUAL_ID,
    i.GENDER = row.GENDER,
    i.AGE = toInteger(row.AGE),
    i.AGE_AT_TUMOUR_RECURRENCE = toInteger(row.AGE_AT_TUMOUR_RECURRENCE),
    i.ETHNICITY = row.ETHNICITY,
    i.ENVIRONMENTAL_VARIABLES = row.ENVIRONMENTAL_VARIABLES,
    i.THERAPY = row.THERAPY,
    i.FAMILY = row.FAMILY,
    i.INDIVIDUAL_REMARK = row.INDIVIDUAL_REMARK;

// ================================
// STEP 4: Verification Queries
// ================================
// Count total nodes imported
MATCH (i:COSMIC_Individual)
RETURN count(i) AS totalIndividuals;

// Sample a few nodes to verify the import
MATCH (i:COSMIC_Individual)
RETURN i
LIMIT 10;

// Check the data types of the age properties
MATCH (i:COSMIC_Individual)
WHERE i.AGE IS NOT NULL
RETURN i.INDIVIDUAL_ID, apoc.meta.type(i.AGE) AS ageType
LIMIT 5;

// ================================
// SCRIPT COMPLETION MESSAGE
// ================================
RETURN "COSMIC Individual import script completed successfully." AS status;