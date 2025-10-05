// convert HAS_GOTERM_CC relationship to HAS_CELLULAR_COMPONENT
MATCH (a)-[r:HAS_GOTERM_CC]->(b)
CALL {
  WITH a, r, b
  CREATE (a)-[r2:HAS_CELLULAR_COMPONENT]->(b)
  SET r2 = properties(r)
  DELETE r
} IN TRANSACTIONS OF 1000 ROWS

// convert HAS_GOTERM_MF relationship to HAS_MOLECULAR_FUNCTION
MATCH (a)-[r:HAS_GOTERM_MF]->(b)
CALL {
  WITH a, r, b
  CREATE (a)-[r2:HAS_MOLECULAR_FUNCTION]->(b)
  SET r2 = properties(r)
  DELETE r
} IN TRANSACTIONS OF 1000 ROWS

// convert HAS_GOTERM_BP relationship to HAS_BIOLOGICAL_PROCESS
MATCH (a)-[r:HAS_GOTERM_BP]->(b)
CALL {
  WITH a, r, b
  CREATE (a)-[r2:HAS_BIOLOGICAL_PROCESS]->(b)
  SET r2 = properties(r)
  DELETE r
} IN TRANSACTIONS OF 1000 ROWS