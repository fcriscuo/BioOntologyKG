// CQL commands to import data from the COSMIC Breakpoints  TSV file

// n.b. Ths CQL should not be executed until the COSMIC Sample data has been imported
//
//      .

// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create a unique constraint on the COSMIC_FUSION_ID property.
CREATE CONSTRAINT cosmic_breakpoint_unique IF NOT EXISTS
FOR (n:COSMIC_Breakpoint) REQUIRE (n.COSMIC_SAMPLE_ID, n.COSMIC_STRUCTURAL_ID) IS UNIQUE;

CREATE INDEX breakpoint_sample_id_index FOR (n:COSMIC_Breakpoint) ON n.COSMIC_SAMPLE_ID;

// ================================
// STEP 2: Delete COSMIC_Fusion nodes if necessary
// ================================
MATCH (n:COSMIC_Breakpoint)
CALL (n) {
    DETACH DELETE n
} IN TRANSACTIONS OF 1000 ROWS;

//Column Headings:
// SAMPLE_NAME	COSMIC_SAMPLE_ID	COSMIC_PHENOTYPE_ID	COSMIC_STRUCTURAL_ID
// MUTATION_TYPE	CHROM_FROM	LOCATION_FROM_MIN	LOCATION_FROM_MAX	STRAND_FROM	CHROM_TO
// LOCATION_TO_MIN	LOCATION_TO_MAX	STRAND_TO	NON_TEMPLATED_INS_SEQ	PUBMED_PMID	COSMIC_STUDY_ID

LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_Breakpoints_v102_GRCh38.tsv' AS row FIELDTERMINATOR '\t'
CALL (row) {
// Create the COSMIC_Gene node
MERGE (b:COSMIC_Breakpoint{ COSMIC_SAMPLE_ID:row.COSMIC_SAMPLE_ID, COSMIC_STRUCTURAL_ID:row.COSMIC_STRUCTURAL_ID})
ON CREATE SET b = row,
        b.LOCATION_FROM_MIN = toInteger(row.LOCATION_FROM_MIN),
        b.LOCATION_FROM_MAX = toInteger(row.LOCATION_FROM_MAX),
        b.LOCATION_TO_MIN = toInteger(row.LOCATION_TO_MIN),
        b.LOCATION_TO_MAX = toInteger(row.LOCATION_TO_MAX)

// If PUBMED_PMID is not null, create a HAS_PUBLICATION relationship
WITH b, row
  // If PUBMED_PMID is not null, create a HAS_PUBLICATION relationship
  OPTIONAL MATCH (p:PubMedArticle {pubmed_id: TRIM(row.PUBMED_PMID)})
  WHERE row.PUBMED_PMID IS NOT NULL
  FOREACH (ignoreMe IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
    MERGE (b)-[:HAS_PUBLICATION]->(p)
  )

// create relationship to sample
WITH b, row
OPTIONAL MATCH (s:COSMIC_Sample {COSMIC_SAMPLE_ID: TRIM(row.COSMIC_SAMPLE_ID)})
  WHERE row.COSMIC_SAMPLE_ID IS NOT NULL
  FOREACH (ignoreMe IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
    MERGE (s)-[:HAS_BREAKPOINT]->(b)
  )

} IN TRANSACTIONS OF 1000 ROWS;