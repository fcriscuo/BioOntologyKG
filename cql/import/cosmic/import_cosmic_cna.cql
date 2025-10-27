// CQL commands to import data from the Cosmic_CNA TSV file

// n.b. Ths CQL should not be executed until the COSMIC Sample, and COSMIC Gene data
//      have been imported.
//      The COSMIC file name is CNA, but the data refers to copy number variations
// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create an index for the COSMIC_SAMPLE_ID
CREATE INDEX cnv_sample_id_index FOR (n:COSMIC_CNV) ON n.COSMIC_SAMPLE_ID;

// TSV file column headings used for node property names
// COSMIC_CNV_ID	COSMIC_GENE_ID	GENE_SYMBOL	COSMIC_SAMPLE_ID	SAMPLE_NAME
// COSMIC_PHENOTYPE_ID	TOTAL_CN	MINOR_ALLELE	MUT_TYPE	COSMIC_STUDY_ID	CHROMOSOME
// GENOME_START	GENOME_STOP


// ================================
// STEP 2: Delete all existing COSMIC_CNV nodes if necessary
// ================================
MATCH (n:COSMIC_CNV)
CALL (n) {
    DETACH DELETE n
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 3: Import CNV data from the TSV file
// ================================
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_CompleteCNA_v102_GRCh38.tsv'
AS row FIELDTERMINATOR '\t'

CALL (row) {
// Create the COSMIC_CNV node
CREATE (n:COSMIC_CNV{ })
SET n = row,
        n.GENOME_START = toInteger(row.GENOME_START),
        n.GENOME_START = toInteger(row.GENOME_STOP)

// create a HAS_COPY_NUMBER_MUTATION relationship from COSMIC_Sample
WITH n, row
OPTIONAL MATCH (s:COSMIC_Sample {COSMIC_SAMPLE_ID: TRIM(row.COSMIC_SAMPLE_ID)})
  WHERE row.COSMIC_SAMPLE_ID IS NOT NULL
  FOREACH (ignoreMe IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
    MERGE (s)-[:HAS_COPY_NUMBER_MUTATION]->(n)
  )

// create a HAS_COSMIC_GENE relationship to COSMIC_Gene
WITH n, row
OPTIONAL MATCH (g:COSMIC_Gene{COSMIC_GENE_ID: TRIM(row.COSMIC_GENE_ID)})
WHERE row.COSMIC_GENE_ID IS NOT NULL
 FOREACH (ignoreMe IN CASE WHEN g IS NOT NULL THEN [1] ELSE [] END |
    MERGE (n)- [:HAS_COSMIC_GENE] ->(g)
)

} IN TRANSACTIONS OF 1000 ROWS;