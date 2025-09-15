//
// Creates a uniqueness constraint on the NCIT_ID property of NCIT_Term nodes.
// This ensures that each NCIT term has a unique identifier.
//
CREATE CONSTRAINT ncit_term_ncit_id IF NOT EXISTS FOR (n:NCIT_Term) REQUIRE n.NCIT_ID IS UNIQUE;

//
// Imports NCIT Thesaurus data from a CSV file into the Neo4j database.
// Use bash command to filter canine-related content from the NCIT.cv file
// head -1 NCIT_sample.csv > NCIT_cleaned.csv
// tail -n +2 NCIT_sample.csv | grep -v -i -E "\b(schnauzer|terrier|retriever|spaniel|poodle|bulldog|beagle)\b" >>
// NCIT_cleaned.csv
//
// This script performs the following actions:
// 1. Delete existing NCIT_Term nodes
// 2. Loads data from the CSV file with headers.
// 3. Processes the data in transactions of 1000 rows for efficiency.
// 4. Creates a node with the label 'NCIT_Term' for each row.
// 5. Maps CSV columns to node properties, converting spaces in property names to underscores.
// 6. Ignores the 'Class ID' column.
// 7. Maps the 'code' column to the 'NCIT_ID' property.
// 8. Splits property values containing '|' into a list of strings.
//
CALL apoc.periodic.iterate(
    'MATCH (n:NCIT_Term) RETURN n',
    'DELETE n',
    {batchSize: 1000, parallel: false}
) YIELD batches, total, timeTaken, committedOperations
RETURN batches, total

LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/GenomicCoreData/NCIT/filtered_NCIT.csv' AS row
CALL {
    WITH row
    CREATE (t: NCIT_Term {NCIT_ID: row.code})
    WITH t, row
    UNWIND [k IN keys(row) WHERE k <> 'code'] AS key
    CALL apoc.create.setProperty(t, replace(key, ' ', '_'),
        CASE
            WHEN row[key] CONTAINS '|' THEN split(row[key], '|')
            ELSE row[key]
        END
    ) YIELD node
    //RETURN count(*) AS processed
    FINISH
} IN TRANSACTIONS OF 1000 ROWS;

