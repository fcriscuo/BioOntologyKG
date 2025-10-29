// COSMIC Complete Targeted Screens Mutant Data Import Script for Neo4j v2025
// This script imports COSMIC Complete Targeted Screens Mutant data.
// n.b.The following nodes must be imported before executing this script
//     COSMIC_Gene
//     COSMIC_Sample
//     COSMIC_Classification
//     COSMIC_Transcript


// ================================
// STEP 1: Clean Existing Data
// ================================
// Detach and delete all existing COSMIC_TargetedScreensMutant nodes if necessary
MATCH (c:COSMIC_TargetedScreensMutant)
CALL (c) {
    DETACH DELETE c
} IN TRANSACTIONS OF 1000 ROWS;

// COSMIC_TargetedScreensMutant_v102_GRCh38.tsv column headers
// GENE_SYMBOL	COSMIC_GENE_ID	TRANSCRIPT_ACCESSION	COSMIC_SAMPLE_ID	SAMPLE_NAME	COSMIC_PHENOTYPE_ID
// GENOMIC_MUTATION_ID	LEGACY_MUTATION_ID	MUTATION_ID	MUTATION_CDS	MUTATION_AA
// MUTATION_DESCRIPTION	MUTATION_ZYGOSITY	LOH	CHROMOSOME	GENOME_START
// GENOME_STOP	STRAND	PUBMED_PMID	COSMIC_STUDY_ID	HGVSP	HGVSC	HGVSG	GENOMIC_WT_ALLELE	GENOMIC_MUT_ALLELE
// MUTATION_SOMATIC_STATUS	POSITIVE_SCREEN

// Step 2 Import data from TSV file
// Change the file name and path as required
LOAD CSV WITH HEADERS
FROM 'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_CompleteTargetedScreensMutant_v102_GRCh38.tsv'
AS row FIELDTERMINATOR '\t'
CALL (row){
CREATE (c:COSMIC_TargetedScreensMutant)
SET c = row,
  c.GENOME_START= toInteger(row.GENOME_START),
  c.GENOME_STOP = toInteger(row.GENOME_STOP),
  c.PUBMED_PMID = TRIM(row.PUBMED_PMID),
  c.HGVSP = toFloat(row.HGVSP),
  c.HGVSC = toFloat(row.HGVSC),
  c.HGVSG = toFloat(row.HGVSG)

// Step 3 Create relationships to existing nodes
// relationship to COSMIC_Gene
WITH c
MATCH (cg:COSMIC_Gene {COSMIC_GENE_ID: c.COSMIC_GENE_ID})
MERGE (c)-[:HAS_COSMIC_GENE]->(cg)

// Create a HAS_SAMPLE relationship with COSMIC_Sample
WITH c
WHERE c.COSMIC_SAMPLE_ID IS NOT NULL
  MERGE (s:COSMIC_Sample{COSMIC_SAMPLE_ID:c.COSMIC_SAMPLE_ID})
  MERGE (c)-[:HAS_SAMPLE]->(s)

//If the CompleteTargetedScreensMutant row specifies a PubMed ID, create a HAS_PUBLICATION relationship
WITH c
WHERE c.PUBMED_PMID IS NOT null
    MERGE (p:Publication {pubmed_id: TRIM(c.PUBMED_PMID)})
    MERGE (c)-[:HAS_PUBLICATION]->(p)

// Create a HAS_PHENOTYPE relation with a COSMIC_Classification node
WITH c
WHERE c.COSMIC_PHENOTYPE_ID IS NOT NULL
  MERGE (cl:COSMIC_Classification {COSMIC_PHENOTYPE_ID:c.COSMIC_PHENOTYPE_ID})
  MERGE (c)-[:HAS_PHENOTYPE]->(cl)

// Create a HAS_TRANSCRIPT  relationship to COSMIC_Transcript
WITH c
WHERE c.TRANSCRIPT_ACCESSION IS NOT NULL
  MERGE (t:COSMIC_Transcript {TRANSCRIPT_ACCESSION: c.TRANSCRIPT_ACCESSION})
  MERGE (c)-[:HAS_TRANSCRIPT]->(t)

// Create a HAS_MUTATION to COSSMIC_MutantCensus
WITH c
  OPTIONAL MATCH (m:COSMIC_MutantCensus {COSMIC_PHENOTYPE_ID: TRIM(c.COSMIC_PHENOTYPE_ID)})
  WHERE c.COSMIC_PHENOTYPE_ID IS NOT NULL
  FOREACH (ignoreMe IN CASE WHEN m IS NOT NULL THEN [1] ELSE [] END |
    MERGE (c)-[:HAS_MUTATION]->(m)
  )

} IN TRANSACTIONS OF 1000 ROWS;