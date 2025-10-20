// CQL commands to import data from the Cosmic_ResistanceMutations (CSV)  TSV file

// n.b. Ths CQL should not be executed until the COSMIC Sample, Cosmic Mutant Censius,
// and COSMIC Classification (i.e Phenotype) data have been imported
//      .


//Column Headings:
// SAMPLE_NAME	COSMIC_SAMPLE_ID	GENE_SYMBOL	COSMIC_GENE_ID	TRANSCRIPT_ACCESSION
// CENSUS_GENE	DRUG_NAME	DRUG_RESPONSE	GENOMIC_MUTATION_ID	LEGACY_MUTATION_ID	MUTATION_ID
// MUTATION_AA	MUTATION_CDS	GENOMIC_WT_ALLELE	GENOMIC_MUT_ALLELE	COSMIC_PHENOTYPE_ID	PUBMED_PMID
// COSMIC_STUDY_ID	MUTATION_ZYGOSITY	CHROMOSOME	GENOME_START	GENOME_STOP	STRAND
// HGVSP	HGVSC	HGVSG	MUTATION_SOMATIC_STATUS


// ================================
// STEP 2: Clean Existing Data
// ================================
// Detach and delete all existing COSMIC_Gene nodes in batches.
MATCH (m:COSMIC_ResistanceMutation)
CALL (m) {
    DETACH DELETE m
} IN TRANSACTIONS OF 1000 ROWS;

/ ================================
// STEP 3: Import Resistance Mutation data from the TSV file
// ================================
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_ResistanceMutations_v102_GRCh38.tsv'
AS row
FIELDTERMINATOR '\t'

WITH row
WHERE row.GENOMIC_MUTATION_ID IS NOT NULL
CALL (row) {

CREATE (n:COSMIC_ResistanceMutation {})
  SET
  n.GENOMIC_MUTATION_ID = TRIM(row.GENOMIC_MUTATION_ID),
  n.SAMPLE_NAME = TRIM(row.SAMPLE_NAME),
  n.COSMIC_SAMPLE_ID = TRIM(row.COSMIC_SAMPLE_ID),
  n.GENE_SYMBOL = TRIM(row.GENE_SYMBOL),
  n.COSMIC_GENE_ID = TRIM(row.COSMIC_GENE_ID),
  n.TRANSCRIPT_ACCESSION = row.TRANSCRIPT_ACCESSION,
  n.CENSUS_GENE = TRIM(row.CENSUS_GENE),
  n.DRUG_NAME = TRIM(row.DRUG_NAME),
  n.DRUG_RESPONSE	= row.DRUG_RESPONSE,
  n.LEGACY_MUTATION_ID = TRIM(row.LEGACY_MUTATION_ID),
  n.MUTATION_ID = TRIM(row.MUTATION_ID),
  n.MUTATION_AA = row.MUTATION_AA,
  n.MUTATION_CDS = row.MUTATION_CDS,
  n.GENOMIC_WT_ALLELE = row.GENOMIC_WT_ALLELE,
  n.GENOMIC_MUT_ALLELE = row.GENOMIC_MUT_ALLELE,
  n.COSMIC_PHENOTYPE_ID = TRIM(row.COSMIC_PHENOTYPE_ID),
  n.PUBMED_PMID = TRIM(row.PUBMED_PMID),
  n.COSMIC_STUDY_ID = TRIM(row.COSMIC_STUDY_ID),
  n.MUTATION_ZYGOSITY = row.MUTATION_ZYGOSITY,
  n.CHROMOSOME = TRIM(row.CHROMOSOME),
  n.GENOME_START = toInteger(row.GENOME_START),
  n.GENOME_STOP = toInteger(row.GENOME_STOP),
  n.STRAND = TRIM(row.STRAND),
  n.HGVSP = toFloat(row.HGVSP),
  n.HGVSC = toFloat(row.HGVSC),
  n.HGVSG = toFloat(row.HGVSG),
  n.MUTATION_SOMATIC_STATUS = TRIM(row.MUTATION_SOMATIC_STATUS)

// Relationships:
// COSMIC_Sample - [HAS_RESISTANCE_MUTATION] -> COSMIC_ResistanceMutation
// COSMIC_ResistanceMutation  -[HAS_PUBLICATION] -> PubMedArticle
// COSMIC_ResistanceMutation -[HAS_COSMIC_GENE] -> COSMIC_GENE
// COSMIC_ResistanceMutation - [HAS_MUTATION] -> COSMIC_MutantCensus
// All the node properties used to create the noted relationships already
// have indexes
//

// If PUBMED_PMID is not null, create a HAS_PUBLICATION relationship
WITH n, row
  // If PUBMED_PMID is not null, create a HAS_PUBLICATION relationship
  OPTIONAL MATCH (p:PubMedArticle {pubmed_id: TRIM(row.PUBMED_PMID)})
  WHERE row.PUBMED_PMID IS NOT NULL
  FOREACH (ignoreMe IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
    MERGE (n)-[:HAS_PUBLICATION]->(p)
  )

// create relationship to sample
WITH n, row
OPTIONAL MATCH (s:COSMIC_Sample {COSMIC_SAMPLE_ID: TRIM(row.COSMIC_SAMPLE_ID)})
  WHERE row.COSMIC_SAMPLE_ID IS NOT NULL
  FOREACH (ignoreMe IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
    MERGE (s)-[:HAS_RESISTANCE_MUTATION]->(n)
  )
// create a relationship to COSMIC_Gene
WITH n, row
OPTIONAL MATCH (g:COSMIC_Gene{COSMIC_GENE_ID: TRIM(row.COSMIC_GENE_ID)})
WHERE row.COSMIC_GENE_ID IS NOT NULL
 FOREACH (ignoreMe IN CASE WHEN g IS NOT NULL THEN [1] ELSE [] END |
    MERGE (n)- [:HAS_COSMIC_GENE] - >(g)
)

// create a relationship to COSMIC_MutantCensus
WITH n,row
OPTIONAL MATCH(m:COSMIC_MutantCensus{GENOMIC_MUTATION_ID: TRIM(row.GENOMIC_MUTATION_ID)})
WHERE row.GENOMIC_MUTATION_ID IS NOT NULL
FOREACH (ignoreMe IN CASE WHEN m IS NOT NULL THEN [1]
ELSE []
END |
MERGE (m) - [:HAS_RESISTANCE_MUTATION] - > (n)
)
} IN TRANSACTIONS OF 1000 ROWS;