// CQL commands to import data from the Cosmic_NonCodingVariants (CNCV)  TSV file

// n.b. Ths CQL should not be executed until the COSMIC Sample and COSMIC Gene data
//      have been imported.


//Column Headings:
// GENE_SYMBOL	COSMIC_GENE_ID	TRANSCRIPT_ACCESSION	MUTATION_NC_ID	SAMPLE_NAME
// COSMIC_SAMPLE_ID	COSMIC_PHENOTYPE_ID	GENOMIC_MUTATION_ID	LEGACY_MUTATION_ID	ZYGOSITY
// CHROMOSOME	GENOME_START	GENOME_STOP	GENOMIC_WT_ALLELE	GENOMIC_MUT_ALLELE
// COSMIC_STUDY_ID	PUBMED_PMID	HGVSG	MUTATION_SOMATIC_STATUS

// Unique property: MUTATION_NC_ID

// Relationships:
// Cosmic_Sample - [HAS_NONCDDING_VARIANT] -> CNCV
// CNCV -[HAS_PUBLICATION] -> PubMedArticle
// CNCV -[HAS_PHENOTYPE] -> CosmicClassification
// Cosmic_Gene -[:HAS_NONCODING_VARIANT] -CNCV

// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create a unique constraint on the TUMOUR_ID property.
CREATE CONSTRAINT cosmic_mutation_nc_id_unique IF NOT EXISTS
FOR (n:COSMIC_NonCodingVariant) REQUIRE n.MUTATION_NC_ID IS UNIQUE;

// ================================
// STEP 2: Delete COSMIC_NonCodingVariant nodes if necessary
// ================================
MATCH (n:COSMIC_NonCodingVariant)
CALL (n) {
    DETACH DELETE n
} IN TRANSACTIONS OF 1000 ROWS;

//================================
// STEP 3: Import NonCoding Variant Data and Create Relationships
// ================================

LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_NonCodingVariants_v102_GRCh38.tsv'
AS row FIELDTERMINATOR '\t'

CALL (row) {

MERGE (n:COSMIC_NonCodingVariant {MUTATION_NC_ID: TRIM(row.MUTATION_NC_ID)})
ON CREATE SET n.COSMIC_GENE_ID = TRIM(row.COSMIC_GENE_ID),
  n.TRANSCRIPT_ACCESSION = row.TRANSCRIPT_ACCESSION,
  n.SAMPLE_NAME = row.SAMPLE_NAME,
  n.COSMIC_SAMPLE_ID = TRIM(row.COSMIC_SAMPLE_ID),
  n.COSMIC_PHENOTYPE_ID = TRIM(row.COSMIC_PHENOTYPE_ID),
  n.GENOMIC_MUTATION_ID = TRIM(row.GENOMIC_MUTATION_ID),
  n.LEGACY_MUTATION_ID = row.LEGACY_MUTATION_ID,
  n.ZYGOSITY = TRIM(row.ZYGOSITY),
  n.CHROMOSOME	= TRIM(row.CHROMOSOME),
  n.GENOME_START = toInteger(row.GENOME_START),
  n.GENOME_STOP = toInteger(row.GENOME_STOP),
  n.GENOMIC_WT_ALLELE = row.GENOMIC_WT_ALLELE,
  n.GENOMIC_MUT_ALLELE = row.GENOMIC_MUT_ALLELE,
  n.COSMIC_STUDY_ID = row.COSMIC_STUDY_ID,
  n.PUBMED_PMID = TRIM(row.PUBMED_PMID),
  n.HGVSG = row.HGVSG,
  n.MUTATION_SOMATIC_STATUS = TRIM(row.MUTATION_SOMATIC_STATUS)

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


// If COSMIC_GENE_ID is not null, create a HAS_NONCODING_VARIANT
WITH n, row

  OPTIONAL MATCH (g:COSMIC_GENE {COSMIC_GENE_ID: TRIM(row.COSMIC_GENE_ID)})
  WHERE row.COSMIC_GENE_ID IS NOT null
  FOREACH (ignoreMe IN CASE WHEN g IS NOT NULL THEN [1] ELSE [] END |
    MERGE (g)-[:HAS_NONCODING_VARIANT]->(n)
  )


// Create a HAS_NONCODING_VARIANT relationship from COSMIC_SAMPLE
WITH n, row
  OPTIONAL MATCH (s:COSMIC_Sample {COSMIC_SAMPLE_ID: TRIM(row.COSMIC_SAMPLE_ID)})
  WHERE row.COSMIC_SAMPLE_ID IS NOT null
  FOREACH (ignoreMe IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
    MERGE(s)-[:HAS_NON_CODING_VARIANT]->(n)
  )

} IN TRANSACTIONS OF 1000 ROWS;


