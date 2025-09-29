// Cypher Query Language (CQL) script to import data from the following DAVID files:
//          DAVID GENE_SYMBOL2DAVID_GENE_NAME.txt
// DAVID data files obtained by requests to: https://davidbioinformatics.nih.gov/knowledgebase/knowledgebaseRequest.jsp
//
//Citations B.T. Sherman, M. Hao, J. Qiu, X. Jiao, M.W. Baseler, H.C. Lane,
//T. Imamichi and W. Chang.
//DAVID: a web server for functional enrichment analysis and functional annotation of
//gene lists (2021 update). Nucleic Acids Research. 23 March 2022. doi:10.1093/nar/gkac194.[PubMed]
//Huang DW, Sherman BT, Lempicki RA.
//Systematic and integrative analysis of large gene lists using DAVID Bioinformatics Resources.
//Nature Protoc. 2009;4(1):44-57.

// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create a unique constraint on the COSMIC_PHENOTYPE_ID property.
CREATE CONSTRAINT david_gene_symbol_unique IF NOT EXISTS
FOR (g:Gene) REQUIRE g.gene_symbol IS UNIQUE;

// ================================
// STEP 2: Clean Existing Data
// ================================
// Detach and delete all existing Gene nodes in batches.
MATCH (g:Gene)
CALL (g) {
    DETACH DELETE g
} IN TRANSACTIONS OF 1000 ROWS;



// ================================
// STEP 3: Import genes from file
// ================================
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/DAVID/david_gene.tsv'
AS row FIELDTERMINATOR '\t'

CALL (row) {
// Create the Gene node
  MERGE(g:Gene{GENE_SYMBOL:row.GENE_SYMBOL})
    ON CREATE SET g.GENE_NAME = row.GENE_NAME
    ON MATCH SET g.GENE_NAME = g.GENE_NAME +","+row.GENE_NAME
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 4: Verification Queries
// ================================
// Count total nodes imported
MATCH (g:Gene)
RETURN count(g) AS totalNodes;

//==================================
// Process DAVID Entrez Gene Summary file
// Add ENTREZ_GENE_SUMMARY as a new Gene node property
//===================================
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/DAVID/david_entrez_gene_summary.tsv'
AS row FIELDTERMINATOR '\t'

CALL (row) {
// MATCH the gene symbols and set a new node property
    MERGE(g:Gene{GENE_SYMBOL:row.GENE_SYMBOL})
    ON MATCH SET g.ENTREZ_GENE_SUMMARY = row.ENTREZ_GENE_SUMMARY
} IN TRANSACTIONS OF 1000 ROWS;

//==================================
// Process DAVID Cytogenetic Location file
// Add ENTREZ_GENE_SUMMARY as a new Gene node property
//===================================
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/DAVID/david_cytogenetic_location.tsv'
AS row FIELDTERMINATOR '\t'

CALL (row) {
// MATCH the gene symbols and set a new node property
    MERGE(g:Gene{GENE_SYMBOL:row.GENE_SYMBOL})
    ON MATCH SET g.CYTOGENETIC_LOCATION = row.CYTOGENETIC_LOCATION
} IN TRANSACTIONS OF 1000 ROWS;


// Create HAS_HGNC relationships from Gene to HGNC based on the GENE_SYMBOL property
MATCH (h:HGNC) WHERE h.symbol IS NOT NULL
WITH h
CALL (h) {
  MATCH (g:Gene) WHERE g.GENE_SYMBOL = h.symbol
  MERGE (g)-[:HAS_HGNC]->(h)
}IN TRANSACTIONS OF 1000 ROWS;
