//Excerpt From
//Knowledge Graphs and LLMs in Action
//Alessandro Negro with Vlastimil Kus, Giuseppe Futia and Fabio Montagna

// Create constraints for PPI nodes
// CREATE CONSTRAINT ncit_term_ncit_id IF NOT EXISTS FOR (n:NCIT_Term) REQUIRE n.NCIT_ID IS UNIQUE;
CREATE CONSTRAINT protein_id IF NOT EXISTS FOR (n:Protein) REQUIRE (n.id) IS UNIQUE;
// Disease node is already defined if HPO has been imported
//CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (n:Disease) REQUIRE (n.id) IS UNIQUE;

// Import the PPI pathways association file
CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/SNAP_biopathways/bio-pathways-associations.csv' AS line RETURN line",
  "WITH trim(line['Associated Gene IDs']) AS proteins,
        trim(line['Disease Name']) AS diseaseName,
        trim(line['Disease ID']) AS diseaseId
   MERGE (d:Disease {id: diseaseId, name: diseaseName})
   WITH d, proteins
   UNWIND split(proteins, ',') AS protein
   WITH d, trim(protein) AS cleanProtein
   WHERE cleanProtein <> ''
   MERGE (p:Protein {id: cleanProtein})
   MERGE (d)-[:ASSOCIATED_WITH]->(p)",
  {batchSize: 100, parallel: false}
);

// Import the PPI network CSV file from https://snap.stanford.edu/pathways/
CALL apoc.periodic.iterate(
  "LOAD CSV FROM 'file:///Volumes/SSD870/data/SNAP_biopathways/bio-pathways-network.csv' AS line RETURN line",
  "MERGE (f:Protein {id: trim(line[0])})
   MERGE (s:Protein {id: trim(line[1])})
   MERGE (f)-[:INTERACTS_WITH]->(s)",
  {batchSize: 100, parallel: false}
);

// Import the PPI disease classifications file: bio-pathways-diseaseclasses.csv
CALL apoc.periodic.iterate(
  "LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/SNAP_biopathways/bio-pathways-diseaseclasses.csv' AS line RETURN line",
  "WITH trim(line['Disease Class']) AS class,
        trim(line['Disease ID']) AS diseaseId
        MATCH(d:Disease {id: diseaseId})
        SET d.class = class",
  {batchSize: 100, parallel: false}
);
