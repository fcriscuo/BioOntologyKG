// CQL script to import COSMIC Mutation Census data into a Neo4j V2035 database
// n.b. The following nodes must be imported before executing this script:
//    COSMIC_Gene
//    COSMIC_Sample
//    COSMIC_Classification
//
// ================================
// STEP 1: Create Unique Constraints
// ================================
//CREATE CONSTRAINT cosmic_mutation_census_unique IF NOT EXISTS
//FOR (m:COSMIC_MutantCensus) REQUIRE  (m.COSMIC_SAMPLE_ID, m.MUTATION_ID) IS UNIQUE;
CREATE INDEX cosmic_mutant_census_genomic_mutation_index FOR(n:COSMIC_MutantCensus) ON n.GENOMIC_MUTATION_ID;
// ================================
// STEP 2: Clean Existing Data
// ================================
// Detach and delete all existing COSMIC_Gene nodes in batches.
MATCH (m:COSMIC_MutantCensus)
CALL (m) {
    DETACH DELETE m
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 3: Import Mutant Census nodes
// ================================

LOAD CSV WITH HEADERS FROM
'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_MutantCensus_v102_GRCh38.tsv' AS row FIELDTERMINATOR '\t'
CALL (row) {
// Create the COSMIC_Gene node
MERGE (g:COSMIC_MutantCensus{ MUTATION_ID:row.MUTATION_ID})
ON CREATE SET g = row,
        g.GENOME_START = toInteger(row.GENOME_START),
        g.GENOME_STOP = toInteger(row.GENOME_STOP)

// Create  a HAS_MUTATION relationship
WITH g, row
MATCH (cg:COSMIC_Gene {COSMIC_GENE_ID: row.COSMIC_GENE_ID})
MERGE (cg)-[:HAS_MUTATION]->(g)

// Create a HAS_MUTATION relationship with COSMIC_Sample
WITH g, row
WHERE row.COSMIC_SAMPLE_ID IS NOT NULL
  MERGE (s:COSMIC_Sample{COSMIC_SAMPLE_ID:row.COSMIC_SAMPLE_ID})
  MERGE (S)-[:HAS_MUTATION]->(g)

//If the Sample row specifies a PubMed ID, create a HAS_PUBLICATION relationship
WITH g, row
WHERE row.PUBMED_PMID IS NOT null
    MERGE (p:Publication {pubmed_id: row.PUBMED_PMID})
    MERGE (g)-[:HAS_PUBLICATION]->(p)

// Create a HAS_PHENOTYPE relation with a COSMIC_Classification node
WITH g,row
WHERE row.PHENOTYPE_ID IS NOT NULL
  MERGE (c:COSMIC_Classification {COSMIC_PHENOTYPE_ID:row.COSMIC_PHENOTYPE_ID})
  MERGE (g)-[:HAS_PHENOTYPE]->(c)

} IN TRANSACTIONS OF 1000 ROWS;
