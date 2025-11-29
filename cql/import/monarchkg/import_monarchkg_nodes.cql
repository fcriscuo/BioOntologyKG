// MonarchKG Node Data Import Script for Neo4j v2025
// This script imports TSV data with dynamic node labels based on the 'category' column
// and handles list conversion for bracketed values
// Only imports human data (NCBITaxon:9606) or records without taxon information

// Load the TSV file and process each row
LOAD CSV WITH HEADERS FROM 'file:///Volumes/SSD870/data/MonarchKG/monarch-kg/monarch-kg_nodes.tsv' AS row
FIELDTERMINATOR '\t'

CALL (row) {
// Filter for human data only: include if in_taxon is null OR contains NCBITaxon:9606
WITH row
WHERE row.in_taxon IS NULL
   OR row.in_taxon = 'NCBITaxon:9606'
   OR row.in_taxon = "['NCBITaxon:9606']"
   OR row.in_taxon CONTAINS 'NCBITaxon:9606'

// Use the full category value as the node label (e.g., "biolink:Gene")
// This preserves the biolink ontology prefix
WITH row,
     row.category AS nodeLabel

// Process list fields (remove quotes and brackets, split by comma)
WITH row, nodeLabel,
     CASE
       WHEN row.xref IS NOT NULL AND row.xref STARTS WITH '[' THEN
         [item IN split(replace(replace(replace(row.xref, '[', ''), ']', ''), "'", ''), ',') | trim(item)]
       WHEN row.xref IS NOT NULL THEN [row.xref]
       ELSE NULL
     END AS xref_list,

     CASE
       WHEN row.synonyms IS NOT NULL AND row.synonyms STARTS WITH '[' THEN
         [item IN split(replace(replace(replace(row.synonyms, '[', ''), ']', ''), "'", ''), ',') | trim(item)]
       WHEN row.synonyms IS NOT NULL THEN [row.synonyms]
       ELSE NULL
     END AS synonyms_list,

     CASE
       WHEN row.synonym IS NOT NULL AND row.synonym STARTS WITH '[' THEN
         [item IN split(replace(replace(replace(row.synonym, '[', ''), ']', ''), "'", ''), ',') | trim(item)]
       WHEN row.synonym IS NOT NULL THEN [row.synonym]
       ELSE NULL
     END AS synonym_list,

     CASE
       WHEN row.exact_synonym IS NOT NULL AND row.exact_synonym STARTS WITH '[' THEN
         [item IN split(replace(replace(replace(row.exact_synonym, '[', ''), ']', ''), "'", ''), ',') | trim(item)]
       WHEN row.exact_synonym IS NOT NULL THEN [row.exact_synonym]
       ELSE NULL
     END AS exact_synonym_list,

     CASE
       WHEN row.broad_synonym IS NOT NULL AND row.broad_synonym STARTS WITH '[' THEN
         [item IN split(replace(replace(replace(row.broad_synonym, '[', ''), ']', ''), "'", ''), ',') | trim(item)]
       WHEN row.broad_synonym IS NOT NULL THEN [row.broad_synonym]
       ELSE NULL
     END AS broad_synonym_list,

     CASE
       WHEN row.narrow_synonym IS NOT NULL AND row.narrow_synonym STARTS WITH '[' THEN
         [item IN split(replace(replace(replace(row.narrow_synonym, '[', ''), ']', ''), "'", ''), ',') | trim(item)]
       WHEN row.narrow_synonym IS NOT NULL THEN [row.narrow_synonym]
       ELSE NULL
     END AS narrow_synonym_list,

     CASE
       WHEN row.related_synonym IS NOT NULL AND row.related_synonym STARTS WITH '[' THEN
         [item IN split(replace(replace(replace(row.related_synonym, '[', ''), ']', ''), "'", ''), ',') | trim(item)]
       WHEN row.related_synonym IS NOT NULL THEN [row.related_synonym]
       ELSE NULL
     END AS related_synonym_list,

     CASE
       WHEN row.same_as IS NOT NULL AND row.same_as STARTS WITH '[' THEN
         [item IN split(replace(replace(replace(row.same_as, '[', ''), ']', ''), "'", ''), ',') | trim(item)]
       WHEN row.same_as IS NOT NULL THEN [row.same_as]
       ELSE NULL
     END AS same_as_list,

     CASE
       WHEN row.subsets IS NOT NULL AND row.subsets STARTS WITH '[' THEN
         [item IN split(replace(replace(replace(row.subsets, '[', ''), ']', ''), "'", ''), ',') | trim(item)]
       WHEN row.subsets IS NOT NULL THEN [row.subsets]
       ELSE NULL
     END AS subsets_list,

     CASE
       WHEN row.provided_by IS NOT NULL AND row.provided_by STARTS WITH '[' THEN
         [item IN split(replace(replace(replace(row.provided_by, '[', ''), ']', ''), "'", ''), ',') | trim(item)]
       WHEN row.provided_by IS NOT NULL THEN [row.provided_by]
       ELSE NULL
     END AS provided_by_list,

     CASE
       WHEN row.has_gene IS NOT NULL AND row.has_gene STARTS WITH '[' THEN
         [item IN split(replace(replace(replace(row.has_gene, '[', ''), ']', ''), "'", ''), ',') | trim(item)]
       WHEN row.has_gene IS NOT NULL THEN [row.has_gene]
       ELSE NULL
     END AS has_gene_list

// Create nodes with dynamic labels using APOC (if available) or MERGE with CASE
CALL apoc.create.node([nodeLabel], {
  id: row.id,
  category: row.category,
  name: row.name,
  description: row.description,
  in_taxon_label: row.in_taxon_label,
  symbol: row.symbol,
  synonyms: synonyms_list,
  xref: xref_list,
  synonym: synonym_list,
  full_name: row.full_name,
  type: row.type,
  has_gene: has_gene_list,
  exact_synonym: exact_synonym_list,
  broad_synonym: broad_synonym_list,
  narrow_synonym: narrow_synonym_list,
  related_synonym: related_synonym_list,
  deprecated: row.deprecated,
  iri: row.iri,
  same_as: same_as_list,
  subsets: subsets_list,
  provided_by: provided_by_list
})  YIELD node
 //RETURN count(node) AS nodesCreated;
} IN TRANSACTIONS OF 1000 ROWS;
// Note: This script uses APOC procedures for dynamic label creation.
// If APOC is not available, you'll need to run separate LOAD CSV commands
// for each node label (category) in your data.