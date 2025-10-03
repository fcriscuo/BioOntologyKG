// Cypher Query Language (CQL) script to import data from the OFFICIAL_GENE_SYMBOL2PUBMED_ID.txt file
// The file downloaded from DAVID is edited to include a header and saved as david_pubmed.tsv
// DAVID data files obtained by requests to: https://davidbioinformatics.nih.gov/knowledgebase/knowledgebaseRequest.jsp
//Citation: B.T. Sherman, M. Hao, J. Qiu, X. Jiao, M.W. Baseler, H.C. Lane,
//T. Imamichi and W. Chang.
//DAVID: a web server for functional enrichment analysis and functional annotation of
//gene lists (2021 update). Nucleic Acids Research. 23 March 2022. doi:10.1093/nar/gkac194.[PubMed]
//Huang DW, Sherman BT, Lempicki RA.
//Systematic and integrative analysis of large gene lists using DAVID Bioinformatics Resources.
//Nature Protoc. 2009;4(1):44-57.
// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create a unique constraint on the Pubmed_id property.
CREATE CONSTRAINT publication_pubmed_id_unique IF NOT EXISTS
FOR (p:Publication) REQUIRE p.pubmed_id IS UNIQUE;
// Create a UNIQUE constraint on the OMIM_DISEASE node
CREATE CONSTRAINT david_omim_unique IF NOT EXISTS
FOR (o:OMIM_Disease) REQUIRE o.OMIM_ID IS UNIQUE;
// Create a UNIQUE constraint for Reactome ID
CREATE CONSTRAINT reactome_id_unique IF NOT EXISTS
FOR (r:Reactome) REQUIRE r.REACTOME_ID IS UNIQUE;

// ================================
// STEP 2: Clean Existing Data
// ================================
// Detach and delete all Publication nodes that are referenced by Gene nodes
MATCH (g:Gene)
CALL (g) {
    MATCH (p:Publcation) WHERE p.pubmed_id = g.PUBMED_ID
    DETACH DELETE p
} IN TRANSACTIONS OF 1000 ROWS;

// OMIM_Disease node
MATCH (o:OMIM_Disease)
CALL (o) {
    DETACH DELETE o
} IN TRANSACTIONS OF 1000 ROWS;


// Reactome nodes
MATCH (r:Reactome)
CALL (r) {
    DETACH DELETE r
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 3: Import pubmed_ids  from the file
// ================================
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/DAVID/david_pubmed_id.tsv'
AS row FIELDTERMINATOR '\t'

CALL (row) {
// Create the Publication node if novel
  MERGE(g:Gene{GENE_SYMBOL:row.GENE_SYMBOL})
  MERGE (p:Publication {pubmed_id: row.PUBMED_ID})
  MERGE (g)-[:HAS_PUBLICATION]->(p)
} IN TRANSACTIONS OF 1000 ROWS;

// Import OMIM data and create a relationship to Gene nodes
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/DAVID/david_omim.tsv'
AS row FIELDTERMINATOR '\t'
CALL (row){
MERGE (o:OMIM_Disease {OMIM_ID:row.OMIM_ID})
  ON CREATE SET o.OMIM_DEFINITION = row.DESCRIPTION
MERGE(g:Gene{GENE_SYMBOL:row.GENE_SYMBOL})
MERGE (g)-[:HAS_OMIM]->(o)
} IN TRANSACTIONS OF 1000 ROWS;


// Import Reactome data and create relationship to Gene nodes
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/DAVID/gene2reactome.tsv'
AS row FIELDTERMINATOR '\t'
CALL (row){
MERGE (r:Reactome{REACTOME_ID:row.REACTOME_ID})
   ON CREATE SET r.REACTOME_PATHWAY =  row.REACTOME_PATHWAY
MERGE(g:Gene{GENE_SYMBOL:row.GENE_SYMBOL})
MERGE (g)-[:HAS_REACTOME_PATHWAY]->(r)
} IN TRANSACTIONS OF 1000 ROWS;

// Gene Ontology
// Establish Gene -[:HAS_GOTERM_BP] -> GoTerm
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/DAVID/david_goterm_bp.tsv'
AS row FIELDTERMINATOR '\t'
CALL (row){
MERGE (go:GOTerm{id:row.GoTermID})
MERGE (g:Gene{GENE_SYMBOL:row.GENE_SYMBOL})
MERGE (g)-[:HAS_GOTERM_BP]->(go)

}IN TRANSACTIONS OF 1000 ROWS;

// Gene Ontology
// Establish Gene -[:HAS_GOTERM_CC] -> GoTerm
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/DAVID/david_goterm_cc.tsv'
AS row FIELDTERMINATOR '\t'
CALL (row){
MERGE (go:GOTerm{id:row.GoTermID})
MERGE (g:Gene{GENE_SYMBOL:row.GENE_SYMBOL})
MERGE (g)-[:HAS_GOTERM_CC]->(go)

}IN TRANSACTIONS OF 1000 ROWS;

// Gene Ontology
// Establish Gene -[:HAS_GOTERM_MF] -> GoTerm
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/DAVID/david_goterm_mf.tsv'
AS row FIELDTERMINATOR '\t'
CALL (row){
MERGE (go:GOTerm{id:row.GoTermID})
MERGE (g:Gene{GENE_SYMBOL:row.GENE_SYMBOL})
MERGE (g)-[:HAS_GOTERM_MF]->(go)

}IN TRANSACTIONS OF 1000 ROWS;

