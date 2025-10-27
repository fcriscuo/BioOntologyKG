// CQL commands to import data from the COSMIC Transcripts  TSV file

// n.b. Ths CQL should not be executed until the COSMIC Gene data has been imported
//
//      .

// ================================
// STEP 1: Create Unique Constraint
// ================================
// Create a unique constraint on the TRANSCRIPT_ACCESSION property.
CREATE CONSTRAINT cosmic_transcript_unique IF NOT EXISTS
FOR (n:COSMIC_Transcript) REQUIRE n.TRANSCRIPT_ACCESSION IS UNIQUE;

CREATE INDEX transcript_gene_id_index FOR (n:COSMIC_Transcript) ON n.COSMIC_GENE_ID;

// ================================
// STEP 2: Delete COSMIC_Transcript nodes if necessary
// ================================
MATCH (n:COSMIC_Transcript)
CALL (n) {
    DETACH DELETE n
} IN TRANSACTIONS OF 1000 ROWS;

//Column Headings:
//TRANSCRIPT_ACCESSION	COSMIC_GENE_ID	LEGACY_GENE_ID	STRAND	BIOTYPE	IS_CANONICAL

// ================================
// STEP 3: Import Transcripts from TSV file
// ================================

LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/COSMIC_rel102/Import/Cosmic_Transcripts_v102_GRCh38.tsv' AS row FIELDTERMINATOR '\t'
CALL (row) {
// Create the COSMIC_Transcript node
// Covert STRAND property to '+ 'and '-' standard
CREATE (t:COSMIC_Transcripts)
SET t = row
  SET t.STRAND = CASE
    WHEN row.STRAND = '1' THEN '+'
    WHEN row.STRAND = '-1' THEN '-'
    ELSE row.STRAND
  END

// create a HAS_COSMIC_GENE relationship to COSMIC_Gene
WITH t, row
OPTIONAL MATCH (g:COSMIC_Gene{COSMIC_GENE_ID: TRIM(row.COSMIC_GENE_ID)})
WHERE row.COSMIC_GENE_ID IS NOT NULL
 FOREACH (ignoreMe IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
    MERGE (t)- [:HAS_COSMIC_GENE] ->(g)
)

} IN TRANSACTIONS OF 1000 ROWS;