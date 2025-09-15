#Excerpt From
#Knowledge Graphs and LLMs in Action
#Alessandro Negro with Vlastimil Kus, Giuseppe Futia and Fabio Montagna
#This material may be protected by copyright.


USE system
CREATE DATABASE hpo
:USE hpo
CREATE CONSTRAINT n10s_unique_uri ON (r:Resource)
ASSERT r.uri IS UNIQUE
CREATE CONSTRAINT n10s_unique_id ON (r:Resource)
ASSERT r.id IS UNIQUE

CALL n10s.graphconfig.init();
CALL n10s.graphconfig.set({ handleVocabUris: "IGNORE" });
CALL n10s.graphconfig.set({ applyNeo4jNaming: True });

CALL n10s.rdf.import.fetch("http://purl.obolibrary.org/obo/hp.owl","RDF/XML");

MATCH (n:Resource)
WHERE n.uri STARTS WITH "http://purl.obolibrary.org/obo/HP"
SET n:Hpo,
    n.id = coalesce(n.id, replace(apoc.text.replace(n.uri,'(.*)obo/',''),'_', ':'));

 CREATE INDEX hpo_id FOR (n:Hpo) ON (n.id);
 CREATE INDEX disease_id FOR (n:Disease) ON (n.id);

#Create disease nodes
LOAD CSV FROM 'https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa' AS row
FIELDTERMINATOR '\t'
WITH row
SKIP 5
MERGE (dis:Resource:Disease {id: row[0]})
ON CREATE SET dis.label = row[1];

#Create the relationships between disease and phenotype
LOAD CSV FROM 'https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa' AS row
FIELDTERMINATOR '\t'
WITH row
SKIP 5
MATCH (dis:Disease)
WHERE dis.id = row[0]
MATCH (phe:Hpo)
WHERE phe.id = row[3]
MERGE (dis)-[:HAS_PHENOTYPIC_FEATURE]->(phe)

#Add properties to the HAS_PHENOTYPE_FEATURE relationships
LOAD CSV FROM 'https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa' AS row
FIELDTERMINATOR '\t'
WITH row
SKIP 5
MATCH (dis:Disease)-[rel:HAS_PHENOTYPIC_FEATURE]->(phe:Hpo)
WHERE phe.id = row[3] and dis.id = row[0]
FOREACH(ignoreMe IN CASE WHEN row[4] is not null THEN [1] ELSE [] END|
SET rel.source = row[4])
FOREACH(ignoreMe IN CASE WHEN row[5] is not null THEN [1] ELSE [] END|
SET rel.evidence = row[5])
FOREACH(ignoreMe IN CASE WHEN row[6] is not null THEN [1] ELSE [] END|
SET rel.onset = row[6])
FOREACH(ignoreMe IN CASE WHEN row[7] is not null THEN [1] ELSE [] END|
SET rel.frequency = row[7])
FOREACH(ignoreMe IN CASE WHEN row[8] is not null THEN [1] ELSE [] END|
SET rel.sex = row[8])
FOREACH(ignoreMe IN CASE WHEN row[9] is not null THEN [1] ELSE [] END|
SET rel.modifier = row[9])
FOREACH(ignoreMe IN CASE WHEN row[10] is not null THEN [1] ELSE [] END|
SET rel.aspect = row[10])

#Enrich the node data to improve readability
MATCH (dis:Disease)-[rel:HAS_PHENOTYPIC_FEATURE]->(phe:Hpo)
SET rel.aspect_name =
CASE
    WHEN rel.aspect = 'P' THEN 'Phenotypic abnormality'
    WHEN rel.aspect = 'I' THEN 'Inheritance'
END,
rel.aspect_description =
CASE
    WHEN rel.aspect = 'P'
    THEN 'Terms with the P aspect are located in
        the Phenotypic abnormality subontology'
    WHEN rel.aspect = 'I'
    THEN 'Terms with the I aspect are from the Inheritance subontology'
END,
rel.evidence_name =
CASE
    WHEN rel.evidence = 'IEA' THEN 'Inferred from electronic annotation'
    WHEN rel.evidence = 'PCS' THEN 'Published clinical study'
    WHEN rel.evidence = 'TAS' THEN 'Traceable author statement'
END,
rel.evidence_description =
CASE
    WHEN rel.evidence = 'IEA' THEN 'Annotations extracted by parsing the Clinical Features sections of the Online Mendelian Inheritance in Man resource are assigned the evidence code "IEA".'
    WHEN rel.evidence = 'PCS' THEN 'PCS is used for used for information extracted from articles in the medical literature. Generally, annotations of this type will include the pubmed id of the published study in the DB_Reference field.'
    WHEN rel.evidence = 'TAS' THEN 'TAS is used for information gleaned from knowledge bases such as OMIM or Orphanet that have derived the information from a published source.'
END,
rel.url =
CASE
    WHEN rel.source STARTS with 'PMID:' THEN 'https://pubmed.ncbi.nlm.nih.gov/' + apoc.text.replace(rel.source, '(.*)PMID:', '')
    WHEN rel.source STARTS with 'OMIM:' THEN 'https://omim.org/entry/' + apoc.text.replace(rel.source, '(.*)OMIM:', '')
END

// Option 1: Using CALL IN TRANSACTIONS (Neo4j 4.4+)
CALL {
    MATCH (dis:Disease)-[rel:HAS_PHENOTYPIC_FEATURE]->(phe:Hpo)
    SET rel.aspect_name =
    CASE
        WHEN rel.aspect = 'P' THEN 'Phenotypic abnormality'
        WHEN rel.aspect = 'I' THEN 'Inheritance'
    END,
    rel.aspect_description =
    CASE
        WHEN rel.aspect = 'P'
        THEN 'Terms with the P aspect are located in the Phenotypic abnormality subontology'
        WHEN rel.aspect = 'I'
        THEN 'Terms with the I aspect are from the Inheritance subontology'
    END,
    rel.evidence_name =
    CASE
        WHEN rel.evidence = 'IEA' THEN 'Inferred from electronic annotation'
        WHEN rel.evidence = 'PCS' THEN 'Published clinical study'
        WHEN rel.evidence = 'TAS' THEN 'Traceable author statement'
    END,
    rel.evidence_description =
    CASE
        WHEN rel.evidence = 'IEA' THEN 'Annotations extracted by parsing the Clinical Features sections of the Online Mendelian Inheritance in Man resource are assigned the evidence code "IEA".'
        WHEN rel.evidence = 'PCS' THEN 'PCS is used for used for information extracted from articles in the medical literature. Generally, annotations of this type will include the pubmed id of the published study in the DB_Reference field.'
        WHEN rel.evidence = 'TAS' THEN 'TAS is used for information gleaned from knowledge bases such as OMIM or Orphanet that have derived the information from a published source.'
    END,
    rel.url =
    CASE
        WHEN rel.source STARTS with 'PMID:' THEN 'https://pubmed.ncbi.nlm.nih.gov/' + apoc.text.replace(rel.source, '(.*)PMID:', '')
        WHEN rel.source STARTS with 'OMIM:' THEN 'https://omim.org/entry/' + apoc.text.replace(rel.source, '(.*)OMIM:', '')
    END
} IN TRANSACTIONS OF 1000 ROWS;

// Option 2: Using APOC Periodic Commit (if APOC is available)
CALL apoc.periodic.commit(
"MATCH (dis:Disease)-[rel:HAS_PHENOTYPIC_FEATURE]->(phe:Hpo)
WHERE rel.aspect_name IS NOT NULL
WITH rel LIMIT $limit
SET rel.aspect_name =
CASE
    WHEN rel.aspect = 'P' THEN 'Phenotypic abnormality'
    WHEN rel.aspect = 'I' THEN 'Inheritance'
END,
rel.aspect_description =
CASE
    WHEN rel.aspect = 'P'
    THEN 'Terms with the P aspect are located in the Phenotypic abnormality subontology'
    WHEN rel.aspect = 'I'
    THEN 'Terms with the I aspect are from the Inheritance subontology'
END,
rel.evidence_name =
CASE
    WHEN rel.evidence = 'IEA' THEN 'Inferred from electronic annotation'
    WHEN rel.evidence = 'PCS' THEN 'Published clinical study'
    WHEN rel.evidence = 'TAS' THEN 'Traceable author statement'
END,
rel.evidence_description =
CASE
    WHEN rel.evidence = 'IEA' THEN 'Annotations extracted by parsing the Clinical Features sections of the Online Mendelian Inheritance in Man resource are assigned the evidence code \"IEA\".'
    WHEN rel.evidence = 'PCS' THEN 'PCS is used for used for information extracted from articles in the medical literature. Generally, annotations of this type will include the pubmed id of the published study in the DB_Reference field.'
    WHEN rel.evidence = 'TAS' THEN 'TAS is used for information gleaned from knowledge bases such as OMIM or Orphanet that have derived the information from a published source.'
END,
rel.url =
CASE
    WHEN rel.source STARTS with 'PMID:' THEN 'https://pubmed.ncbi.nlm.nih.gov/' + apoc.text.replace(rel.source, '(.*)PMID:', '')
    WHEN rel.source STARTS with 'OMIM:' THEN 'https://omim.org/entry/' + apoc.text.replace(rel.source, '(.*)OMIM:', '')
END
RETURN count(rel)",
{limit: 10000}
);

// Option 3: Manual batching with SKIP and LIMIT
// Run this query multiple times, incrementing the SKIP value
MATCH (dis:Disease)-[rel:HAS_PHENOTYPIC_FEATURE]->(phe:Hpo)
WITH rel
SKIP 0 LIMIT 10000  // Change SKIP to 10000, 20000, 30000, etc. for subsequent runs
SET rel.aspect_name =
CASE
    WHEN rel.aspect = 'P' THEN 'Phenotypic abnormality'
    WHEN rel.aspect = 'I' THEN 'Inheritance'
END,
rel.aspect_description =
CASE
    WHEN rel.aspect = 'P'
    THEN 'Terms with the P aspect are located in the Phenotypic abnormality subontology'
    WHEN rel.aspect = 'I'
    THEN 'Terms with the I aspect are from the Inheritance subontology'
END,
rel.evidence_name =
CASE
    WHEN rel.evidence = 'IEA' THEN 'Inferred from electronic annotation'
    WHEN rel.evidence = 'PCS' THEN 'Published clinical study'
    WHEN rel.evidence = 'TAS' THEN 'Traceable author statement'
END,
rel.evidence_description =
CASE
    WHEN rel.evidence = 'IEA' THEN 'Annotations extracted by parsing the Clinical Features sections of the Online Mendelian Inheritance in Man resource are assigned the evidence code "IEA".'
    WHEN rel.evidence = 'PCS' THEN 'PCS is used for used for information extracted from articles in the medical literature. Generally, annotations of this type will include the pubmed id of the published study in the DB_Reference field.'
    WHEN rel.evidence = 'TAS' THEN 'TAS is used for information gleaned from knowledge bases such as OMIM or Orphanet that have derived the information from a published source.'
END,
rel.url =
CASE
    WHEN rel.source STARTS with 'PMID:' THEN 'https://pubmed.ncbi.nlm.nih.gov/' + apoc.text.replace(rel.source, '(.*)PMID:', '')
    WHEN rel.source STARTS with 'OMIM:' THEN 'https://omim.org/entry/' + apoc.text.replace(rel.source, '(.*)OMIM:', '')
END
RETURN count(rel) as updated_relationships;

// Option 4: Batch by Disease nodes (if you have many diseases)
MATCH (dis:Disease)
WITH collect(dis) as diseases
CALL {
    WITH diseases
    UNWIND diseases as disease
    MATCH (disease)-[rel:HAS_PHENOTYPIC_FEATURE]->(phe:Hpo)
    SET rel.aspect_name =
    CASE
        WHEN rel.aspect = 'P' THEN 'Phenotypic abnormality'
        WHEN rel.aspect = 'I' THEN 'Inheritance'
    END,
    rel.aspect_description =
    CASE
        WHEN rel.aspect = 'P'
        THEN 'Terms with the P aspect are located in the Phenotypic abnormality subontology'
        WHEN rel.aspect = 'I'
        THEN 'Terms with the I aspect are from the Inheritance subontology'
    END,
    rel.evidence_name =
    CASE
        WHEN rel.evidence = 'IEA' THEN 'Inferred from electronic annotation'
        WHEN rel.evidence = 'PCS' THEN 'Published clinical study'
        WHEN rel.evidence = 'TAS' THEN 'Traceable author statement'
    END,
    rel.evidence_description =
    CASE
        WHEN rel.evidence = 'IEA' THEN 'Annotations extracted by parsing the Clinical Features sections of the Online Mendelian Inheritance in Man resource are assigned the evidence code "IEA".'
        WHEN rel.evidence = 'PCS' THEN 'PCS is used for used for information extracted from articles in the medical literature. Generally, annotations of this type will include the pubmed id of the published study in the DB_Reference field.'
        WHEN rel.evidence = 'TAS' THEN 'TAS is used for information gleaned from knowledge bases such as OMIM or Orphanet that have derived the information from a published source.'
    END,
    rel.url =
    CASE
        WHEN rel.source STARTS with 'PMID:' THEN 'https://pubmed.ncbi.nlm.nih.gov/' + apoc.text.replace(rel.source, '(.*)PMID:', '')
        WHEN rel.source STARTS with 'OMIM:' THEN 'https://omim.org/entry/' + apoc.text.replace(rel.source, '(.*)OMIM:', '')
    END
} IN TRANSACTIONS OF 1000 ROWS;

# Sample queries
MATCH path=(dis:Disease)-[:HAS_PHENOTYPIC_FEATURE]->(phe:Hpo)
WHERE dis.id = "OMIM:222100"
RETURN path

MATCH (phe:Hpo)
WHERE phe.label = "Growth delay"
OR phe.label = "Large knee"
OR phe.label = "Sensorineural hearing impairment"
OR phe.label = "Pruritus"
OR phe.label = "Type I diabetes mellitus"
WITH phe
MATCH path=(dis:Disease)-[:HAS_PHENOTYPIC_FEATURE]->(phe)
UNWIND dis as nodes
RETURN dis.id as disease_id,
dis.label as disease_name,
collect(phe.label) as features,
count(nodes) as num_of_features
ORDER BY num_of_features DESC, disease_name
LIMIT 5


MATCH (p:Hpo)<-[:SUBCLASSOF*1..3]-(n:Hpo)
WHERE p.id = "HP:0000818"
RETURN p,n

MATCH (cat:Hpo {label: "Abnormality of the endocrine system"})
CALL n10s.inference.nodesInCategory(cat, {
    inCatRel: "HAS_PHENOTYPIC_FEATURE", subCatRel: "SUBCLASSOF"})
YIELD node as dis
MATCH (dis)-[:HAS_PHENOTYPIC_FEATURE]->(phe:Hpo)
RETURN dis.label as disease, collect(DISTINCT phe.label) as features
ORDER BY size(features) ASC, disease
SKIP 200
LIMIT 5