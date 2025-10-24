// CQL commands to import data from the Cosmic_CompleteDifferentialMethylation TSV file

// n.b. Ths CQL should not be executed until the COSMIC Sample, and COSMIC Gene data
//      have been imported.
//

// Column headings used for node property names
// COSMIC_GENE_ID	COSMIC_STUDY_ID	COSMIC_SAMPLE_ID	SAMPLE_NAME	FRAGMENT_ID	CHROMOSOME
// POSITION	STRAND	GENE_SYMBOL	METHYLATION	AVG_BETA_VALUE_NORMAL	BETA_VALUE	TWO_SIDED_P_VALUE	COSMIC_PHENOTYPE_ID

// ================================
//  Delete existing COSMIC_MethDiff nodes if necessary
// ================================
MATCH (n:COSMIC_DiffMeth)
CALL (n) {
    DETACH DELETE n
} IN TRANSACTIONS OF 1000 ROWS;


// ================================
//  Import data for new COSMIC_DiffMeth nodes from TSV file
// ================================

LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_CompleteDifferentialMethylation_v102_GRCh38.tsv'
AS row FIELDTERMINATOR '\t'

CALL (row) {
// Create the COSMIC_DiffMeth node
CREATE (n:COSMIC_DiffMeth{ })
SET n = row,
        n.AVG_BETA_VALUE_NORMAL = toFloat(row.AVG_BETA_VALUE_NORMAL),
        n.BETA_VALUE = toFloat(row.BETA_VALUE),
        n.TWO_SIDED_P_VALUE = toFloat(row.TWO_SIDED_P_VALUE)

// create relationship from COSMIC_Sample
WITH n, row
OPTIONAL MATCH (s:COSMIC_Sample {COSMIC_SAMPLE_ID: TRIM(row.COSMIC_SAMPLE_ID)})
  WHERE row.COSMIC_SAMPLE_ID IS NOT NULL
  FOREACH (ignoreMe IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
    MERGE (s)-[:HAS_DIFF_METYLATION]->(n)
  )

// create a relationship to COSMIC_Gene
WITH n, row
OPTIONAL MATCH (g:COSMIC_Gene{COSMIC_GENE_ID: TRIM(row.COSMIC_GENE_ID)})
WHERE row.COSMIC_GENE_ID IS NOT NULL
 FOREACH (ignoreMe IN CASE WHEN g IS NOT NULL THEN [1] ELSE [] END |
    MERGE (n)- [:HAS_COSMIC_GENE] ->(g)
)

} IN TRANSACTIONS OF 1000 ROWS;