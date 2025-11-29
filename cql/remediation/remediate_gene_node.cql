// Gene Node Consolidation: Merge Gene into biolink:Gene
// This query consolidates redundant Gene nodes into biolink:Gene nodes
// by copying properties and migrating relationships

MATCH (g:Gene)

CALL (g) {

WITH g
MATCH (bg:`biolink:Gene`)
  WHERE g.GENE_SYMBOL = bg.symbol

// Step 1: Copy properties from Gene to biolink:Gene (if not null)
FOREACH (_ IN CASE WHEN g.ENTREZ_GENE_SUMMARY IS NOT NULL THEN [1]
ELSE []
END |
SET bg.ENTREZ_GENE_SUMMARY = g.ENTREZ_GENE_SUMMARY
)
FOREACH (_ IN CASE WHEN g.gene_summary_embedding IS NOT NULL THEN [1]
ELSE []
END |
SET bg.gene_summary_embedding = g.gene_summary_embedding
)
FOREACH (_ IN CASE WHEN g.CYTOGENIC_LOCATION IS NOT NULL THEN [1]
ELSE []
END |
SET bg.CYTOGENIC_LOCATION = g.CYTOGENIC_LOCATION
)

// Step 2: Migrate COSMIC_GENE relationships
WITH g, bg
OPTIONAL MATCH (cg:COSMIC_GENE)- [r1:HAS_GENE] - >(g)
FOREACH (_ IN CASE WHEN cg IS NOT NULL THEN [1]
ELSE []
END |
MERGE (cg)- [:HAS_GENE] - >(bg)
)

// Step 3: Migrate HAS_PUBLICATION relationships (avoid duplicates)
WITH g, bg
OPTIONAL MATCH (g)- [r2:HAS_PUBLICATION] - >(p:PubMedArticle)
WITH g, bg, p
WHERE p IS NOT null
AND NOT exists((bg)- [:HAS_PUBLICATION] - >(p))
FOREACH (_ IN [1] |
MERGE (bg)- [:HAS_PUBLICATION] - >(p)
)

// Step 4: Delete the redundant Gene node and its relationships
WITH g
DETACH DELETE g
} IN TRANSACTIONS OF 1000 ROWS
//RETURN count(*) AS nodes_consolidated;