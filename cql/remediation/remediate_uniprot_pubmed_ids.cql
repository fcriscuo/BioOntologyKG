// The PubMed IDs imported from the UniProt data contain character values with leading blanks.
// This results in PubMedArticle nodes whose pubmed_id property doesn't link to the OubMed ID values
// found in related nodes (e.g. Gene)
// The following Cypher commands correct this error in the UniProt and PubMedArticle nodes
// Find and fix PubMedArticle nodes with whitespace in pubmed_id

// Preview nodes that will be updated
MATCH (p:PubMedArticle)
WHERE p.pubmed_id <> trim(p.pubmed_id)
RETURN p.pubmed_id as original,
       trim(p.pubmed_id) as trimmed,
       count(*) as count
LIMIT 20

// trim the pubmed_id property
MATCH (p:PubMedArticle)
WHERE p.pubmed_id <> trim(p.pubmed_id)
SET p.pubmed_id = trim(p.pubmed_id)
RETURN count(p) as nodes_updated

// Process UniProt nodes in batches of 10,000
CALL apoc.periodic.iterate(
    "MATCH (u:UniProt) WHERE u.PubMed_ID IS NOT NULL RETURN u",
    "SET u.PubMed_ID = [id IN u.PubMed_ID WHERE trim(id) <> '' | trim(id)]",
    {batchSize: 10000, parallel: false}
)
YIELD batches, total
RETURN batches, total