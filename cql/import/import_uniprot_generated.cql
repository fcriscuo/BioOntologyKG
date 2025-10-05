// UniProt Data Import Script for Neo4j v5
// This script imports specified columns from a UniProt TSV file.

// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create a unique constraint on the Entry property for UniProt nodes.
CREATE CONSTRAINT uniprot_entry_unique IF NOT EXISTS
FOR (u:UniProt) REQUIRE u.Entry IS UNIQUE;

// ================================
// STEP 2: Clean Existing Data
// ================================
// Detach and delete existing UniProt nodes in batches to avoid memory issues.
MATCH (u:UniProt)
CALL (u) {
    DETACH DELETE u
} IN TRANSACTIONS OF 1000 ROWS;

// ================================
// STEP 3: Import UniProt Data
// ================================
// Load data from the TSV file, process specified columns, and create nodes.
//:auto USING PERIODIC COMMIT 1000
 LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/UniProt/current/uniprotkb_primary_2025_08_28.tsv' AS row FIELDTERMINATOR '\t'
CALL (row) {
    MERGE (u:UniProt {Entry: row.Entry})
    SET u.Entry_Name = row.`Entry Name`,
        u.Protein_names = CASE WHEN row.`Protein names` CONTAINS ';' THEN split(row.`Protein names`, ';') ELSE row.`Protein names` END,
        u.Gene_names = CASE WHEN row.`Gene names` CONTAINS ';' THEN split(row.`Gene names`, ';') ELSE row.`Gene names` END,
        u.Keywords = CASE WHEN row.Keywords CONTAINS ';' THEN split(row.Keywords, ';') ELSE row.Keywords END,
        u.Gene_Ontology_IDs = CASE WHEN row.`Gene Ontology IDs` CONTAINS ';' THEN split(row.`Gene Ontology IDs`, ';') ELSE row.`Gene Ontology IDs` END,
       // u.Interacts_with = CASE WHEN row.`Interacts with` CONTAINS ';' THEN split(row.`Interacts with`, ';') ELSE row.`Interacts with` END,
        u.Interacts_with = split(row.`Interacts with`, ';'),
        //u.PubMed_ID = CASE WHEN row.`PubMed ID` CONTAINS ';' THEN split(row.`PubMed ID`, ';') ELSE row.`PubMed ID` END,
        u.PubMed_ID = CASE
            WHEN row.`PubMed ID` CONTAINS ';'
            THEN [id IN split(row.`PubMed ID`, ';') WHERE trim(id) <> '' | trim(id)]
            ELSE trim(row.`PubMed ID`)
        END,
        u.AlphaFoldDB = CASE WHEN row.AlphaFoldDB CONTAINS ';' THEN split(row.AlphaFoldDB, ';') ELSE row.AlphaFoldDB END,
        u.BioGRID = CASE WHEN row.BioGRID CONTAINS ';' THEN split(row.BioGRID, ';') ELSE row.BioGRID END,
        u.CORUM = CASE WHEN row.CORUM CONTAINS ';' THEN split(row.CORUM, ';') ELSE row.CORUM END,
        u.IntAct = CASE WHEN row.IntAct CONTAINS ';' THEN split(row.IntAct, ';') ELSE row.IntAct END,
        u.MINT = CASE WHEN row.MINT CONTAINS ';' THEN split(row.MINT, ';') ELSE row.MINT END,
        u.STRING = CASE WHEN row.STRING CONTAINS ';' THEN split(row.STRING, ';') ELSE row.STRING END,
        u.Ensembl = CASE WHEN row.Ensembl CONTAINS ';' THEN split(row.Ensembl, ';') ELSE row.Ensembl END,
        u.HGNC = CASE WHEN row.HGNC CONTAINS ';' THEN split(row.HGNC, ';') ELSE row.HGNC END,
        u.DisGeNET = CASE WHEN row.DisGeNET CONTAINS ';' THEN split(row.DisGeNET, ';') ELSE row.DisGeNET END,
        u.Reactome = CASE WHEN row.Reactome CONTAINS ';' THEN split(row.Reactome, ';') ELSE row.Reactome END,
        u.Bgee = CASE WHEN row.Bgee CONTAINS ';' THEN split(row.Bgee, ';') ELSE row.Bgee END
} IN TRANSACTIONS OF 1000 ROWS;
   
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/UniProt/current/uniprotkb_primary_2025_08_28.tsv' AS row FIELDTERMINATOR '\t'
CALL (row) {
    MATCH (u:UniProt {Entry: row.Entry})
    SET u.Interacts_with: split(row.`Interacts with`, ';')
} IN TRANSACTIONS OF 1000 ROWS;
   

// ================================
// STEP 4: Verification Queries
// ================================
// Count the total number of UniProt nodes imported.
MATCH (u:UniProt)
RETURN count(u) AS totalUniProtNodes;

// Sample a few nodes to verify the import and property mapping.
MATCH (u:UniProt)
WHERE u.Entry IS NOT NULL
RETURN u
LIMIT 5;

// Check a node with a list property to verify the split operation.
MATCH (u:UniProt)
WHERE u.`Gene_Ontology_IDs` IS NOT NULL AND size(u.`Gene_Ontology_IDs`) > 1
RETURN u.Entry, u.`Gene_Ontology_IDs`
LIMIT 5;

// Verify that trailing semicolons were removed from single-value fields.
MATCH (u:UniProt)
WHERE u.HGNC IS NOT NULL AND NOT u.HGNC ENDS WITH ';'
RETURN u.Entry, u.HGNC
LIMIT 5;


// ================================
// SCRIPT COMPLETION MESSAGE
// ================================
RETURN "UniProt data import script completed successfully. Verify results using the queries above." AS status;
