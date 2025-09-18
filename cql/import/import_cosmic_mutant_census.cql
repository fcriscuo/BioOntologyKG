// ================================
// STEP 1: Create Unique Constraints
// ================================
//CREATE CONSTRAINT cosmic_mutation_census_unique IF NOT EXISTS
//FOR (m:COSMIC_Mutant_Census) REQUIRE  (m.COSMIC_SAMPLE_ID, m.MUTATION_ID) IS UNIQUE;

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
MERGE (g:COSMIC_Mutant_Census{COSMIC_SAMPLE_ID:row.COSMIC_SAMPLE_ID, COSMIC_PHENOTYPE_ID:row.COSMIC_PHENOTYPE_ID,
    GENOMIC_MUTATION_ID:row.GENOMIC_MUTATION_ID, MUTATION_ID:row.MUTATION_ID,
     COSMIC_GENE_ID:row.COSMIC_GENE_ID})
ON CREATE SET g = row,
        g.GENOME_START = toInteger(row.GENOME_START),
        g.GENOME_STOP = toInteger(row.GENOME_STOP)

// Create  a HAS_COSMIC_GENE relationship
WITH g, row
MATCH (cg:COSMIC_Gene {COSMIC_GENE_ID: row.COSMIC_GENE_ID})
MERGE (g)-[:HAS_COSMIC_GENE]->(cg)

// Create a HAS_MUTATION relationship with COSMIC_Sample
WITH g, row
MATCH (s:COSMIC_Sample{COSMIC_SAMPLE_ID:g.COSMIC_SAMPLE_ID})
MERGE (S)-[:HAS_MUTATION]->(g)

//If the Sample row specifies a PubMed ID, create a HAS_PUBLICATION relationship
WITH g, row
WHERE row.PUBMED_PMID IS NOT null
    MERGE (p:Publication {pubmed_id: row.PUBMED_PMID})
    MERGE (g)-[:HAS_PUBLICATION]->(p)

} IN TRANSACTIONS OF 1000 ROWS;
