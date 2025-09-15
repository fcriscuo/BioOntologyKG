// ============================================================================
// Neo4j Cypher Queries for PubMed Vector Search
// ============================================================================

// 1. CREATE VECTOR INDEX
// Creates a vector index for abstract embeddings with cosine similarity
CREATE VECTOR INDEX pubmed_abstract_embeddings IF NOT EXISTS
FOR (p:Publication)
ON p.abstractEmbedding
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 384,  // Change based on your embedding model
        `vector.similarity_function`: 'cosine'
    }
}

// 2. CREATE PUBLICATION NODE WITH EMBEDDING
// Example of creating a publication node with embedding and metadata
MERGE (p:Publication {pmid: "33747033"})
SET p.title = "Example biomedical research article",
    p.abstract = "This study investigates molecular mechanisms...",
    p.abstractEmbedding = [0.1, 0.2, 0.3, ...],  // Your embedding vector
    p.firstAuthor = "John Smith",
    p.journalTitle = "Nature Biotechnology",
    p.volume = "39",
    p.issue = "3",
    p.yearPublished = "2021",
    p.doi = "10.1038/s41587-021-00123-4",
    p.createdAt = datetime()
RETURN p.pmid as pmid

// 3. VECTOR SIMILARITY SEARCH
// Search for similar abstracts using vector index
CALL db.index.vector.queryNodes('pubmed_abstract_embeddings', 5, [0.1, 0.2, 0.3, ...])  // Your query embedding
YIELD node, score
RETURN node.pmid as pmid,
       node.title as title,
       node.abstract as abstract,
       node.firstAuthor as firstAuthor,
       node.journalTitle as journalTitle,
       node.volume as volume,
       node.issue as issue,
       node.yearPublished as yearPublished,
       node.doi as doi,
       score
ORDER BY score DESC

// 4. FILTERED SIMILARITY SEARCH
// Search with additional filters (e.g., by year or journal)
CALL db.index.vector.queryNodes('pubmed_abstract_embeddings', 10, [0.1, 0.2, 0.3, ...])
YIELD node, score
WHERE toInteger(node.yearPublished) >= 2020
  AND node.journalTitle CONTAINS "Nature"
RETURN node.pmid as pmid,
       node.title as title,
       node.firstAuthor as firstAuthor,
       node.journalTitle as journalTitle,
       node.yearPublished as yearPublished,
       score
ORDER BY score DESC
LIMIT 5

// 5. CREATE AUTHOR RELATIONSHIPS
// Create author nodes and link to publications
MATCH (p:Publication {pmid: "33747033"})
UNWIND [
    {firstName: "John", lastName: "Smith", initials: "JS", position: 1},
    {firstName: "Jane", lastName: "Doe", initials: "JD", position: 2}
] as author_data
MERGE (a:Author {
    firstName: author_data.firstName,
    lastName: author_data.lastName,
    initials: author_data.initials
})
MERGE (a)-[:AUTHORED {position: author_data.position}]->(p)

// 6. CREATE MESH TERM RELATIONSHIPS
// Create MeSH term nodes and relationships
MATCH (p:Publication {pmid: "33747033"})
UNWIND [
    {descriptor: "Neoplasms", majorTopic: true},
    {descriptor: "Cell Proliferation", majorTopic: false}
] as mesh_data
MERGE (m:MeshTerm {descriptor: mesh_data.descriptor})
MERGE (p)-[:HAS_MESH_TERM {majorTopic: mesh_data.majorTopic}]->(m)

// 7. BATCH CREATE PUBLICATIONS
// Create multiple publications at once
UNWIND $publications as pub_data
MERGE (p:Publication {pmid: pub_data.pmid})
SET p.title = pub_data.title,
    p.abstract = pub_data.abstract,
    p.abstractEmbedding = pub_data.abstractEmbedding,
    p.firstAuthor = pub_data.firstAuthor,
    p.journalTitle = pub_data.journalTitle,
    p.volume = pub_data.volume,
    p.issue = pub_data.issue,
    p.yearPublished = pub_data.yearPublished,
    p.doi = pub_data.doi,
    p.createdAt = datetime()

// 8. FIND SIMILAR PUBLICATIONS BY PMID
// Find publications similar to a specific PMID
MATCH (source:Publication {pmid: "33747033"})
CALL db.index.vector.queryNodes('pubmed_abstract_embeddings', 6, source.abstractEmbedding)
YIELD node, score
WHERE node.pmid <> source.pmid  // Exclude the source publication
RETURN node.pmid as pmid,
       node.title as title,
       node.firstAuthor as firstAuthor,
       node.journalTitle as journalTitle,
       node.yearPublished as yearPublished,
       score
ORDER BY score DESC
LIMIT 5

// 9. DATABASE STATISTICS
// Get statistics about your publication database
MATCH (p:Publication)
RETURN count(p) as totalPublications,
       count(p.abstractEmbedding) as publicationsWithEmbeddings,
       min(toInteger(p.yearPublished)) as earliestYear,
       max(toInteger(p.yearPublished)) as latestYear,
       count(DISTINCT p.journalTitle) as uniqueJournals

// 10. AUTHOR COLLABORATION NETWORK
// Find authors who have collaborated (published together)
MATCH (a1:Author)-[:AUTHORED]->(p:Publication)<-[:AUTHORED]-(a2:Author)
WHERE a1 <> a2
RETURN a1.firstName + " " + a1.lastName as author1,
       a2.firstName + " " + a2.lastName as author2,
       count(p) as collaborations,
       collect(p.pmid) as sharedPublications
ORDER BY collaborations DESC
LIMIT 20

// 11. JOURNAL PUBLICATION COUNTS
// Count publications by journal
MATCH (p:Publication)
WHERE p.journalTitle IS NOT NULL
RETURN p.journalTitle as journal,
       count(p) as publicationCount,
       min(toInteger(p.yearPublished)) as earliestPublication,
       max(toInteger(p.yearPublished)) as latestPublication
ORDER BY publicationCount DESC
LIMIT 20

// 12. MESH TERM ANALYSIS
// Find most common MeSH terms
MATCH (p:Publication)-[r:HAS_MESH_TERM]->(m:MeshTerm)
RETURN m.descriptor as meshTerm,
       count(p) as publicationCount,
       sum(CASE WHEN r.majorTopic THEN 1 ELSE 0 END) as majorTopicCount
ORDER BY publicationCount DESC
LIMIT 20

// 13. DELETE VECTOR INDEX
// Remove the vector index if needed
DROP INDEX pubmed_abstract_embeddings

// 14. HYBRID SEARCH (Vector + Text)
// Combine vector similarity with text search
CALL db.index.vector.queryNodes('pubmed_abstract_embeddings', 20, [0.1, 0.2, 0.3, ...])
YIELD node, score
WHERE node.title CONTAINS "cancer"
   OR node.abstract CONTAINS "cancer"
RETURN node.pmid as pmid,
       node.title as title,
       node.firstAuthor as firstAuthor,
       node.journalTitle as journalTitle,
       node.yearPublished as yearPublished,
       score
ORDER BY score DESC
LIMIT 10

// 15. UPDATE EMBEDDING FOR EXISTING PUBLICATION
// Update embedding for a publication that already exists
MATCH (p:Publication {pmid: "33747033"})
SET p.abstractEmbedding = [0.1, 0.2, 0.3, ...]  // New embedding vector
RETURN p.pmid, "Embedding updated" as status