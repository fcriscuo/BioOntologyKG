// CQL script to import MonarchKG NCIT term nodes
// detach and delete any existing nodes
CALL {
    MATCH (n:NCIT_Term)
    DETACH DELETE n
} IN TRANSACTIONS OF 1000 ROWS;

// STEP 2: Create a uniqueness constraint
// Ensures that the NCIT_ID  property for each NCIT_Term node is unique.
CREATE CONSTRAINT IF NOT EXISTS FOR (n:NCIT_Term) REQUIRE n.NCIT_ID IS UNIQUE;

// read the TSV file for NCIT_Term nodes
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/MonarchKG/ncit_kgx_tsv_nodes.tsv' AS row FIELDTERMINATOR '\t'

CALL (row){
  CREATE (n:NCIT_Term:biolink:NamedThing)
    SET n.NCIT_ID = row.id,
    n.name = row.name,
    n.description =row.description,
    n.synonym = split(row.synonym,'|')
} IN TRANSACTIONS OF 2000 ROWS