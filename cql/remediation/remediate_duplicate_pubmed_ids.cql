CALL {
    MATCH (p:PubMedArticle)
    WITH p.pubmed_id AS pmid, collect(p) AS nodes
    WHERE size(nodes) > 1
    WITH pmid, nodes
    LIMIT 10000  // Process 1000 duplicate groups at a time

    WITH pmid, nodes[0] AS keeper, nodes[1..] AS to_delete
    UNWIND to_delete AS dup
    DETACH DELETE dup
    RETURN count(dup) AS deleted
}
RETURN sum(deleted) AS total_deleted

CREATE CONSTRAINT pubmed_id_unique IF NOT EXISTS
FOR (p:PubMedArticle) REQUIRE p.pubmed_id IS UNIQUE