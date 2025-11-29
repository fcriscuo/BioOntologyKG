// MonarchKG NCIT Relationship Import Script for Neo4j v2025
// Import relationships between NCIT_Term nodes from TSV file

// Step 1: Load and create relationships in batches of 1000
CALL apoc.periodic.iterate(
  // First query: Load the TSV data
  "LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/MonarchKG/ncit_kgx_tsv_edges.tsv'
   AS row FIELDTERMINATOR '\t'
   RETURN row",

  // Second query: Create relationships
  "MATCH (source:NCIT_Term {NCIT_ID: row.subject})
   MATCH (target:NCIT_Term {NCIT_ID: row.object})
   WITH source, target, row
   CALL apoc.create.relationship(
     source,
     row.predicate,
     {knowledge_source: row.knowledge_source},
     target
   ) YIELD rel
   RETURN count(rel) as relationships_created",

  // Configuration: batch size and parallel execution
  {batchSize: 1000, parallel: false}
)
YIELD batches, total, errorMessages
RETURN batches, total, errorMessages;

// Alternative approach without APOC (if APOC is not available):
// Note: This approach is less efficient but works without plugins

/*
:auto LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/MonarchKG/ncit_kgx_tsv_edges.tsv'
AS row FIELDTERMINATOR '\t'
CALL {
  WITH row
  MATCH (source:NCIT_Term {NCIT_ID: row.subject})
  MATCH (target:NCIT_Term {NCIT_ID: row.object})
  // Handle dynamic relationship types using CASE or individual queries
  // Note: Neo4j doesn't support fully dynamic relationship types in pure Cypher
  // You'll need to enumerate expected relationship types
  WITH source, target, row
  WHERE row.predicate = 'biolink:subclass_of'
  CREATE (source)-[r:SUBCLASS_OF {knowledge_source: row.knowledge_source}]->(target)
  RETURN count(r) as created
} IN TRANSACTIONS OF 1000 ROWS
RETURN count(*) as total_processed;
*/

// Verification queries after import:

// Check total relationships created
MATCH ()-[r]->()
WHERE r.knowledge_source IS NOT NULL
RETURN count(r) as total_relationships;

// Check relationship type distribution
MATCH ()-[r]->()
WHERE r.knowledge_source IS NOT NULL
RETURN type(r) as relationship_type, count(r) as count
ORDER BY count DESC;

// Sample relationships
MATCH (source:NCIT_Term)-[r]->(target:NCIT_Term)
WHERE r.knowledge_source IS NOT NULL
RETURN source.NCIT_ID, type(r), target.NCIT_ID, r.knowledge_source
LIMIT 10;