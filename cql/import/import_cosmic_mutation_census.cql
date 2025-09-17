// ================================
// STEP 1: Create Unique Constraints
// ================================
CREATE CONSTRAINT cosmic_mutation_census_unique IF NOT EXISTS
FOR (m:COSMIC_Mutant_Census) REQUIRE  (m.COSMIC_SAMPLE_ID, m.MUTATION_ID) IS UNIQUE;

// ================================
// STEP 2: Clean Existing Data
// ================================
// Detach and delete all existing COSMIC_Gene nodes in batches.
MATCH (m:COSMIC_Mutant_Census)
CALL (m) {
    DETACH DELETE m
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 3: Import Mutant Census nodes
// ================================

LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_MutantCensus_v102_GRCh38.tsv' AS row FIELDTERMINATOR '\t'
CALL (row) {
// Create the COSMIC_Gene node
MERGE (g:COSMIC_Mutant_Census{gene_sample_id:row.COSMIC_SAMPLE_ID, cosmic_pphenotype_id:row.COSMIC_PHENOTYPE_ID,
    genomic_mutation_id:row.GENOMIC_MUTATION_ID, mutation_id:row.MUTATION_ID,
     cosmic_gene_id:row.COSMIC_GENE_ID})
ON CREATE SET g = row,
        g.GENOME_START = toInteger(row.GENOME_START),
        g.GENOME_STOP = toInteger(row.GENOME_STOP),
        g.gene_symbol = row.GENE_SYMBOL,
        g.transcript_Accession = row.TRANSCRIPT_ACCESSION,
        g.sample_name = row.SAMPLE_NAME,
        g.legacy_mutation_id = row.LEGACY_MUTATION_ID,
        g.mutation_cds = row.MUTATION_CDS,
        g.mutation_aa = row.MUTATION_AA,
        g.mutation_description = row.MUTATION_DESCRIPTION,
        g.mutation_zygosity = row.MUTATION_ZYGOSITY,
        g.loh = row.LOH,
        g.chromosome = row.CHROMOSOME,
        g.strand = row.STRAND,
        g.pubmed_id = row.PUBMED_PMID,
        g.cosmic_study_id = row.COSMIC_STUDY_ID,
        g.hgvsp = row.HGVSP,
        g.hgvsc = row.HGVSC,
        g.genomic_wt_allele = row.GENOMIC_WT_ALLELE,
        g.genomic_mut_allele = row.GENOMIC_MUT_ALLELE,
        g.mutation_somatic_status=row.GENOMIC_MUT_ALLELE

// Create  a HAS_COSMIC_GENE relationship
WITH g, row
MATCH (cg:COSMIC_Gene {COSMIC_GENE_ID: row.COSMIC_GENE_ID})
MERGE (g)-[:HAS_COSMIC_GENE]->(cg)

// Create a HAS_MUTATION relationship with COSMIC_Sample
WITH g, row
MATCH (s:COSMIC_SAMPLE{COSMIC_SAMPLE_ID:row.COSMIC_SAMPLE_ID})
MERGE (S)-[:HAS_MUTATION]->(g)

//If the Sample row specifies a PubMed ID, create a HAS_PUBLICATION relationship
WITH g, row
WHERE g.PUBMED_PMID IS NOT null AND size(g.PUBMED_PMID) > 0
    MERGE (p:Publication {pubmed_id: g.PUBMED_PMID})
    MERGE (g)-[:HAS_PUBLICATION]->(p)

} IN TRANSACTIONS OF 1000 ROWS;
