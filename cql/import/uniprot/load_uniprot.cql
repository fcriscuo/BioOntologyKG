// UniProt Entry
//Entry	Reviewed	Entry Name	Protein names	Gene Names	Organism	Gene Ontology (biological process)	G
//Gene Ontology (cellular component)	Gene Ontology (molecular function)	Gene Ontology IDs	Gene Ontology (GO)	
//Interacts with	PubMed ID	BioGRID	CORUM	STRING	IntAct	Reactome	HGNC


CALL apoc.periodic.iterate('
CALL apoc.load.csv("/Volumes/SSD870/data/UniProt/current/uniprotkb_primary_2025_08_28.tsv", { header:true, sep:"TAB",
mapping:{
     `Protein names`: {array:true, arraySep:";"},
     `Gene Names`: {array:true, arraySep:";"},
     `Interacts with`: {array:true, arraySep:";"},
     `Gene Ontology IDs`: {array:true, arraySep:";"},
     `PubMed ID`: {array:true, arraySep:";"},
     `HGNC`: {array:true, arraySep:";"},
     `Reactome`: {array:true, arraySep:";"}
     }
}
)
YIELD lineNo, map AS nodeRecord RETURN nodeRecord, lineNo
','
WHERE NOT nodeRecord.`Entry` IS NULL
MERGE (uni:UniProtEntry {entry_id: nodeRecord.Entry})
SET  uni.entry_name =nodeRecord.`Entry Name`,
  uni.uniprot_emtry_url = replace("https://www.uniprot.org/uniprotkb/XXXX/entry","XXXX",uni.entry_id),
  uni.protein_names = nodeRecord.`Protein names`,
  uni.gene_names = nodeRecord.`Gene Names`,
  uni.gene_symbol = nodeRecord.`Gene Names (primary)`,
  uni.interacts_with = nodeRecord.`Interacts with`,
  uni.gene_ontology_ids = nodeRecord.`Gene Ontology IDs`,
  uni.pubmed_ids=nodeRecord.`PubMed ID`,
  uni.reactome_id = nodeRecord.`Reactome`,
  uni.biogrid_ids = nodeRecord.`BioGRID`,
  uni.corum_ids = nodeRecord.`CORUM`,
  uni.string_ids = nodeRecord.`STRING`,
  uni.intact_ids = nodeRecord.`IntAct`,
  uni.hgnc_ids = nodeRecord.`HGNC`[0]
',
{batchSize:1000, iterateList:true});

// set the gene_symbol property for a UniProt entry based on its
// related HGNC
MATCH (uni:UniProtEntry) WHERE uni.hgnc_ids  IS NOT NULL
MATCH(hgnc:HGNC) WHERE hgnc.hgnc_id = uni.hgnc_ids[0]
SET uni.gene_symbol = hgnc.gene_symbol
;

// AlphaFold URL
MATCH (uni:UniProtEntry) WHERE uni.alpha_fold_ids  IS NOT NULL
SET uni.alpha_fold_urls = []
WITH uni
UNWIND uni.alpha_folds_ids as alpha_id
WITH alpha_id, uni
SET uni.alpha_fold_urls = uni.alpha_fold_urls + genomiccore.resolveAlphaFoldUrl(alpha_id)
;

//dbSnp URLs
MATCH (uni:UniProtEntry) WHERE uni.dbSNPs IS NOT NULL
SET uni.dbSnpUrlList = []
WITH uni
UNWIND uni.dbSnps AS snp
SET uni.dbSnpList = uni.dbSnpList + genomiccore.resolvedbSnpUrl(snp)
;

// UniProtEntry -> UniProtEntry relationship (self-interaction)
MATCH (uni:UniProtEntry) WHERE uni.interacts_with IS NOT NULL
UNWIND uni.interacts_with AS target_id
WITH uni, target_id
WHERE target_id IS NOT NULL AND target_id <> ''
MATCH (target:UniProtEntry) WHERE  target.entry_id = target_id
WITH target, uni
WHERE NOT (target)-[:INTERACTS_WITH]->(uni)
MERGE (uni)-[r:INTERACTS_WITH]->(target)
;

// Supplement GenomicEntity's collection of Publication/PubMed relationships
// Create Publication placeholder nodes if Publication is novel
MATCH (uni:UniProtEntry) WHERE uni.pubmed_ids <> ' '
UNWIND uni.pubmed_ids AS pubmed_id
WITH pubmed_id, uni
  WHERE pubmed_id IS NOT NULL AND pubmed_id <> ''
MERGE (p:Publication:PubMed{pub_id:toInteger(ltrim(pubmed_id))})
  ON CREATE
  SET p.url= genomiccore.resolvePubmedUrl(toString(p.pub_id)),
  p.needs_properties=true, p.needs_references=true
WITH p, uni
MATCH (ge:GenomicEntity) WHERE ge.gene_symbol = uni.gene_symbol
MERGE (ge)-[r:HAS_PUBLICATION]->(p)
;
