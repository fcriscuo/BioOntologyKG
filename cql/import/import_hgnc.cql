// Cypher script to import HUGO Gene Nomenclature Committee (HGNC) data from a TSV file into Neo4j v5.

// STEP 1: Clean up existing HGNC data
// Before importing, detach and delete all existing HGNC nodes to ensure a clean slate.
// This operation is performed in batches to avoid memory issues.
CALL {
    MATCH (n:HGNC)
    DETACH DELETE n
} IN TRANSACTIONS OF 1000 ROWS;

// STEP 2: Create a uniqueness constraint
// Ensures that the hgnc_id for each HGNC node is unique.
CREATE CONSTRAINT IF NOT EXISTS FOR (g:HGNC) REQUIRE g.hgnc_id IS UNIQUE;

// STEP 3: Load HGNC data from TSV file
// This query loads the data in batches of 1000 rows.
// It processes the properties dynamically:
// - Ignores any columns where the name starts with "date".
// - Splits values containing a pipe '|' into a list of strings.
// - Sets the remaining columns as properties on the new :HGNC node.
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/HGNC/hgnc_complete_set.tsv' AS row FIELDTERMINATOR '\t'
    
    // Filter out columns that start with 'date'
    WITH row, [key IN keys(row) WHERE NOT key STARTS WITH 'date'] AS validKeys
    
    // Create a map of properties, splitting pipe-delimited strings into lists
    WITH row, apoc.map.fromPairs(
        [key IN validKeys |
            [key,
             CASE
                 WHEN row[key] CONTAINS '|' THEN split(row[key], '|')
                 ELSE row[key]
             END
            ]
        ]
    ) AS properties
    
    // Create the HGNC node and set its properties
    CREATE (h:HGNC)
    SET h += properties
} IN TRANSACTIONS OF 1000 ROWS;

MATCH (h:HGNC)
RETURN count(h) as totalHGNCNodes;

// Sample a few nodes to verify import
MATCH (h:HGNC)
RETURN h
LIMIT 5;

// Check for nodes with list properties (pipe-delimited values)
MATCH (h:HGNC)
WHERE any(prop IN keys(h) WHERE apoc.meta.type(h[prop]) = 'LIST OF STRING')
RETURN h
LIMIT 3;

// Verify constraint is working
SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties, options
WHERE 'HGNC' IN labelsOrTypes;

// ================================
// SCRIPT COMPLETION MESSAGE
// ================================
RETURN "HGNC data import completed successfully. Verify results using the queries above." as status;
