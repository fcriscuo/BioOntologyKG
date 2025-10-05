// Process UniProt nodes in batches of 10,000
CALL apoc.periodic.iterate(
    "MATCH (u:UniProt) WHERE u.Gene_Ontology_IDs IS NOT NULL RETURN u",
    "SET u.Gene_Ontology_IDs = [id IN u.Gene_Ontology_IDs WHERE trim(id) <> '' | trim(id)]",
    {batchSize: 10000, parallel: false}
)
YIELD batches, total
RETURN batches, total