// Create a HAS_GENE relationship from COSMIC_Gene to Gene
MATCH(cg:COSMIC_Gene)
CALL (cg) {
  WITH cg
  MATCH (g:Gene) WHERE cg.GENE_SYMBOL = g.GENE_SYMBOL
  MERGE (cg)-[:HAS_GENE]->(g)
} WITH TRANSACTIONS OF 1000 ROWS