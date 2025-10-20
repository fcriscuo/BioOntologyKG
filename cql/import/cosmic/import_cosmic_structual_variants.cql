// CQL commands to import data from the Cosmic_StructuralVariants (CSV)  TSV file

// n.b. Ths CQL should not be executed until the COSMIC Sample and COSMIC Classification (i.e Phenotype) data
//      have been imported.


//Column Headings:
//SAMPLE_NAME	COSMIC_SAMPLE_ID	COSMIC_PHENOTYPE_ID	COSMIC_STRUCTURAL_ID	MUTATION_TYPE
//DESCRIPTION	PUBMED_PMID	COSMIC_STUDY_ID	ID_STRUC_GEN	CHROMOSOME_FROM	CHROMOSOME_TO
//LOCATION_FROM_MIN	LOCATION_FROM_MAX	LOCATION_TO_MIN	LOCATION_TO_MAX	STRAND_FROM	STRAND_TO

// Unique property: COSMIC_STRUCTURAL_ID

// Relationships:
// Cosmic_Sample - [HAS_STRUCTURAL_VARIANT] -> CSV
// CSV -[HAS_PUBLICATION] -> PubMedArticle
// CSV -[HAS_PHENOTYPE] -> CosmicClassification


// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create a unique constraint on the TUMOUR_ID property.
CREATE CONSTRAINT cosmic_structual_variant_id_unique IF NOT EXISTS
FOR (n:COSMIC_StructuralVariant) REQUIRE n.COSMIC_STRUCTURAL_ID IS UNIQUE;

// ================================
// STEP 2: Delete COSMIC_NonCodingVariant nodes if necessary
// ================================
MATCH (n:COSMIC_StructuralVariant)
CALL (n) {
    DETACH DELETE n
} IN TRANSACTIONS OF 1000 ROWS;

//================================
// STEP 3: Import Structural Variant Data and Create Relationships
// ================================

LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_StructuralVariants_v102_GRCh38.tsv'
AS row FIELDTERMINATOR '\t'

CALL (row) {

MERGE (n:COSMIC_StructuralVariant {COSMIC_STRUCTURAL_ID : TRIM(row.COSMIC_STRUCTURAL_ID)})
ON CREATE SET n.SAMPLE_NAME = TRIM(row.SAMPLE_NAME),
  n.COSMIC_SAMPLE_ID = row.COSMIC_SAMPLE_ID,
  n.COSMIC_PHENOTYPE_ID = TRIM(row.COSMIC_PHENOTYPE_ID),
  n.MUTATION_TYPE = row.MUTATION_TYPE,
  n.DESCRIPTION = row.DESCRIPTION,
  n.PUBMED_PMID = TRIM(row.PUBMED_PMID),
  n.COSMIC_STUDY_ID = TRIM(row.COSMIC_STUDY_ID),
  n.ID_STRUC_GEN = toInteger(row.ID_STRUC_GEN),
  n.CHROMOSOME_FROM = TRIM(row.CHROMOSOME_FROM),
  n.CHROMOSOME_TO = TRIM(row.CHROMOSOME_TO),
  n.LOCATION_FROM_MIN = toInteger(row.LOCATION_FROM_MIN),
  n.LOCATION_FROM_MAX = toInteger(row.LOCATION_FROM_MAX),
  n.LOCATION_TO_MIN = toInteger(row.LOCATION_TO_MIN),
  n.LOCATION_TO_MAX = toInteger(row.LOCATION_TO_MAX),
  n.STRAND_FROM = TRIM(row.STRAND_FROM),
  n.STRAND_TO = TRIM(row.STRAND_TO)

// If PUBMED_PMID is not null, create a HAS_PUBLICATION relationship
WITH n, row

  // If PUBMED_PMID is not null, create a HAS_PUBLICATION relationship
  OPTIONAL MATCH (p:PubMedArticle {pubmed_id: TRIM(row.PUBMED_PMID)})
  WHERE row.PUBMED_PMID IS NOT NULL
  FOREACH (ignoreMe IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
    MERGE (n)-[:HAS_PUBLICATION]->(p)
  )

WITH n, row
/// Create a HAS_PHENOTYPE relationship
  OPTIONAL MATCH (c:COSMIC_Classification {COSMIC_PHENOTYPE_ID: TRIM(row.COSMIC_PHENOTYPE_ID)})
  WHERE row.COSMIC_PHENOTYPE_ID IS NOT NULL
  FOREACH (ignoreMe IN CASE WHEN c IS NOT NULL THEN [1] ELSE [] END |
    MERGE (n)-[:HAS_PHENOTYPE]->(c)
  )

// Create a HAS_STRUCTURAL_VARIANT relationship from COSMIC_SAMPLE
WITH n, row
  OPTIONAL MATCH (s:COSMIC_Sample {COSMIC_SAMPLE_ID: TRIM(row.COSMIC_SAMPLE_ID)})
  WHERE row.COSMIC_SAMPLE_ID IS NOT null
  FOREACH (ignoreMe IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
    MERGE(s)-[:HAS_STRUCTURAL_VARIANT]->(n)
  )

} IN TRANSACTIONS OF 1000 ROWS;
