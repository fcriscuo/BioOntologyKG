// COSMIC Classification Paper Data Import Script for Neo4j v2025
// This script imports COSMIC classification paper data, creates nodes,
// and establishes relationships with COSMIC Classification.
// n.b. The classification data is keyed on COSMIC_PHENOTYPE_PAPER_ID
// The following nodes must be imported before executing this script:
//        COSMIC_Classification

// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create a unique constraint on the COSMIC_PHENOTYPE_PAPER_ID property.
CREATE CONSTRAINT cosmic_classification_paper_id_unique IF NOT EXISTS
FOR (p:COSMIC_ClassificationPaper) REQUIRE p.COSMIC_PHENOTYPE_PAPER_ID  IS UNIQUE;

// ================================
// STEP 2: Clean Existing Data
// ================================
// Detach and delete all existing COSMIC_ClassificationPaper nodes in batches if necessary
MATCH (p:COSMIC_ClassificationPaper)
CALL (p) {
    DETACH DELETE p
} IN TRANSACTIONS OF 1000 ROWS;

// TSV ClassificationPaper column headers
// COSMIC_PHENOTYPE_PAPER_ID	COSMIC_PHENOTYPE_ID	PAPER_PRIMARY_SITE	PAPER_SITE_SUBTYPE_1
// PAPER_SITE_SUBTYPE_2	PAPER_SITE_SUBTYPE_3PAPER_PRIMARY_HISTOLOGY	PAPER_HISTOLOGY_SUBTYPE_1
// PAPER_HISTOLOGY_SUBTYPE_2	PAPER_HISTOLOGY_SUBTYPE_3


// ================================
// STEP 3: Import Data and Create Relationships
// ================================
// Load data from the TSV file, and create COSMIC_ClassificationPaper nodes
// Change the file name and path as required
LOAD CSV WITH HEADERS
FROM 'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_ClassificationPaper_v102_GRCh38.tsv'
AS row FIELDTERMINATOR '\t'
CALL (row){
MERGE (p:COSMIC_ClassificationPaper {COSMIC_PHENOTYPE_PAPER_ID: TRIM(row.COSMIC_PHENOTYPE_PAPER_ID)})
ON CREATE SET p.PAPER_PRIMARY_SITE = TRIM(row.PAPER_PRIMARY_SITE),
  p.PAPER_PRIMARY_SITE = row.PAPER_PRIMARY_SITE,
  p.PAPER_SITE_SUBTYPE_1 = row.PAPER_SITE_SUBTYPE_1,
  p.PAPER_SITE_SUBTYPE_2 = row.PAPER_SITE_SUBTYPE_2,
  p.PAPER_SITE_SUBTYPE_3 = row.PAPER_SITE_SUBTYPE_3,
  p.PAPER_PRIMARY_HISTOLOGY = row.PAPER_PRIMARY_HISTOLOGY,
  p.PAPER_HISTOLOGY_SUBTYPE_1 = row.PAPER_HISTOLOGY_SUBTYPE_1,
  p.PAPER_HISTOLOGY_SUBTYPE_2 = row.PAPER_HISTOLOGY_SUBTYPE_2,
  p.PAPER_HISTOLOGY_SUBTYPE_3 = row.PAPER_HISTOLOGY_SUBTYPE_3

// Create a relationship from COSMIC_Classificationß
 WITH p, row
OPTIONAL MATCH (c:COSMIC_Classification{COSMIC_PHENOTYPE_ID: TRIM(row.COSMIC_PHENOTYPE_ID)})
WHERE row.COSMIC_PHENOTYPE_ID IS NOT NULL
 FOREACH (ignoreMe IN CASE WHEN c IS NOT NULL THEN [1] ELSE [] END |
    MERGE (c)- [:HAS_CLASSIFICATION_PAPER] ->(p)
)

} IN TRANSACTIONS OF 1000 ROWS;