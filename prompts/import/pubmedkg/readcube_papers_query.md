# ReadCube Papers Query script Prompt

## Introduction
I have a repository of approximately 3000 scientific papers managed by ReadCube's Papers
application. I also have a subscription to a major scientific publisher's journals. That
subscription allows me to download thirty (30) papers per month. I want to avoid downloading
a paper that I already have in my repository. I need a utility that will allow me to quickly
determine if a paper that I am interested in, is already in my local repository. The papers in
my repository have been cataloged as PubMedArticle nodes in a local Neo4j v2025 database.
These papers can be distinguished from other PubMedArticle nodes by a NOT NULL value for the
node's local-url property.
As a Neo4j v2025 and Python 3 coding export, I would like you to generate a Python 3 script
that will query the local Neo4j v2025 database for information about a PubMedArticle node representing
a paper in the Paper's repository. 
Please adhere to the following specifications

## Specifications: 

<ol>
<li>The script will prompt the user for a property that identifies a single paper</li>
<li> A paper can be identified by one of three (3) PubMedArticle properties:
<ol>
<li>The paper's PubMed ID which is persisted as the pubmed_id property in a PubMedArticle node</li>
<li> The paper's DOI value which is persisted as the doi value in a PubMedArticle node</li>
<li> The paper's title which can  is persisted as the title property in a PubMedArticle node</li>
</ol>
</li>
<li> The format of a user's input should determine how the PubMedArticle nodes are queried 
<ol>
<li>If the input is numeric, it represents a PubMed ID (e.g. 19308067)</li>
<li>If the input starts with "10." it represents a DOI (e.g. 10.1038/nrc2622) </li>
<li>If the input is alphanumeric it represents an article's complete or partial title
(e.g. 'Metastasis: from dissemination to organ-specific colonization' or 'Metastasis'</li>
</ol>
</li>
<li>Since searching for a title using a substring will usually return more than one article, 
the script should limit its response to ten (10) articles.</li>
<li>When the query identifies one or more articles, the script's output should consist of:
<ol>
<li>The article's PubMed ID</li>
<li>The article's Journal name</li>
<li>The article's complete title</li>
</ol>
</li>
<li>If the query does not find a PubMedArticle, the response should indicate that the paper is not
in the repository</li>
<li>The request/response cycle should run as a loop until the user enters 'STOP'</li>
<li> The Neo4j credentials, NEO_4J_USER, NEO4J_PASSWORD, NEO4J_URI, and NEO4J_DATABASE, should be
obtained using system environment variables</li>
</ol>