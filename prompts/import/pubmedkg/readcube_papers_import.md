## Intrduction
Hello. I have a local ReadCube Papers repository on my desktop that contains approximately 3000 scientific and
papers. 
I would like to import metadata about these papers into my Neo4j v2025 oncology knowledge graph. 
Metadata about the individual papers can be exported to a file in.bib format.
An example of the metadata for an article is included below.
Some of the papers in the repository are awaiting curation. 
Metadata for curated papers should not imported since the PubMed IDs have not been established yet.
Each paper in my Papers repository should map to a new or existing PubMedArticle node 
in the Neo4j database. As Neo4j v2025 and Python expert, I would appreciate it, if you would generate a Python
script to process the .bib file and import curated data info the local Neo4j v2025 database. Please
adhere to the following specifications.

## Specifications
<ol>
<li> The Neo4j credentials, NEO_4J_USER, NEO4J_PASSWORD, NEO4J_URI, and NEO4J_DATABASE, should be
obtained using system environment variables</li>
<li>The user should be prompted for the location of the .bib file</li>
<li>Labeled data from each @article clause should be mapped to novel Neo4j PubMedArticle nodes with the
following properties (.bib file property to PubMedArticle property):
<ol>
<li>year -> publication_date</li>
<li>title -> title</li>
<li>author -> authors</li>
<li>journal -> journal</li>
<li>doi -> doi</li>
<li>>pmid -> pubmed_id</li>
<li>pmcid -> pmcid_url</li>
<li>abstract -> abstract</li>
<li>pages -> pages</li>
<li>number -> number</li>
<li>volume -> volume</li>
<li>local-url -> local-url</li>
</ol>
</li>
<li>If the @article's first amd unlabeled property is "undefined",
the article has not been curated and should be skipped For example:
@article{undefined, 
title = {{Nature Reviews Genetics-2016-Lessons from non-canonical splicing\_1.pdf}}, 
author = {}, 
local-url = {file://localhost/Volumes/SSD870/Users/fcriscuo/Documents/Papers%20Library/Nature%20Reviews%20Genetics-2016-Lessons%20from%20non-canonical%20splicing_1_2.pdf}
}</li>
<li>All properties should be processed as text strings</li>
<li>If the .bib data does not include a pmid property, the article should be ignored</li>
<li>The author property in the .bib file should be imported as a Neo4j list</li>
<li>All brace characters (i.e. '{' , '}', '{{', and '}}' ) should be removed</li>
<li>.bib file properties not listed above should be ignored</li>
<li>The .bip pmcid property should be imported to a PubMedArticle pmcid_url property as a URL
consisting of a fixed component: 'https://pmc.ncbi.nlm.nih.gov/articles/' followed by the pmcid
property value. 
For example pmcid=PMC6367327  maps to pmcid_url: https://pmc.ncbi.nlm.nih.gov/articles/PMC6367327</li>
<li>If the .bib article entry's pmid property matches the pubmed_id property of an existing PubMedArticle node,
only the pmcid, pages, number, volume, and local-url properties should be imported to update 
the existing PubMedArticle node</li>
<li>The ReadCube Papers application exports metadata for its entire repository. Thus as more articles are added 
or curated, articles that have already been imported will appear in during the import process. The Python script 
should determine if a PubMedArticle node need to be created or updated for a .bib article by chacking if the
PubMedArticle node's local-url property is not null. If so, the article can ne ignored.</li>
</ol>

## Sample ReadCube Papers .bib entry
The following data represents a Papers article that has been curated and should be imported into the Neo4j database:
@article{10.1038/nrm3742, 
year = {2014}, 
title = {{A day in the life of the spliceosome}}, 
author = {Matera, A. Gregory and Wang, Zefeng}, 
journal = {Nature Reviews Molecular Cell Biology}, 
issn = {1471-0072}, 
doi = {10.1038/nrm3742}, 
pmid = {24452469}, 
pmcid = {PMC4060434}, 
abstract = {{Spliceosomal snRNAs are transcribed from specialized promoters, which recruit RNA polymerase II cofactors that aid in proper 3′ end maturation of these non-polyadenylated transcripts.Like most non-coding RNAs, small nuclear RNAs (snRNAs) use cognate antisense elements to interact with their nucleic acid targets via base pairing.Assembly of functional small nuclear ribonucleoproteins (snRNPs) involves a series of non-functional intermediates that are often sequestered in subcellular compartments that are distinct from their sites of action.snRNP function requires multiple protein partners (such as DExD/H helicases or WD box proteins) the roles of which may include modulating RNA structure or tethering an enzyme.snRNPs recognize specific sequences in pre-mRNAs and assemble into the spliceosome in a stepwise manner. The splicing reaction itself is catalysed by U6/U2 snRNA complex that resembles a self-splicing ribozyme.Alternative splicing is typically regulated by multiple cis-elements and trans-factors, which form complex interaction networks that may provide a great deal of regulatory plasticity.Pre-mRNA splicing can be regulated throughout the entire spliceosomal assembly pathway, although the early steps are the main stages of regulation. Spliceosomal snRNAs are transcribed from specialized promoters, which recruit RNA polymerase II cofactors that aid in proper 3′ end maturation of these non-polyadenylated transcripts. Like most non-coding RNAs, small nuclear RNAs (snRNAs) use cognate antisense elements to interact with their nucleic acid targets via base pairing. Assembly of functional small nuclear ribonucleoproteins (snRNPs) involves a series of non-functional intermediates that are often sequestered in subcellular compartments that are distinct from their sites of action. snRNP function requires multiple protein partners (such as DExD/H helicases or WD box proteins) the roles of which may include modulating RNA structure or tethering an enzyme. snRNPs recognize specific sequences in pre-mRNAs and assemble into the spliceosome in a stepwise manner. The splicing reaction itself is catalysed by U6/U2 snRNA complex that resembles a self-splicing ribozyme. Alternative splicing is typically regulated by multiple cis-elements and trans-factors, which form complex interaction networks that may provide a great deal of regulatory plasticity. Pre-mRNA splicing can be regulated throughout the entire spliceosomal assembly pathway, although the early steps are the main stages of regulation. The tight regulation of each step of spliceosome assembly from small nuclear RNAs and associated proteins requires coordination between distinct cellular compartments. This in turn dictates where and when alternative splicing occurs and is vital for normal gene expression control. One of the most amazing findings in molecular biology was the discovery that eukaryotic genes are discontinuous, with coding DNA being interrupted by stretches of non-coding sequence. The subsequent realization that the intervening regions are removed from pre-mRNA transcripts via the activity of a common set of small nuclear RNAs (snRNAs), which assemble together with associated proteins into a complex known as the spliceosome, was equally surprising. How do cells coordinate the assembly of this molecular machine? And how does the spliceosome accurately recognize exons and introns to carry out the splicing reaction? Insights into these questions have been gained by studying the life cycle of spliceosomal snRNAs from their transcription, nuclear export and re-import to their dynamic assembly into the spliceosome. This assembly process can also affect the regulation of alternative splicing and has implications for human disease.}}, 
pages = {108--121}, 
number = {2}, 
volume = {15}, 
local-url = {file://localhost/Volumes/SSD870/Users/fcriscuo/Documents/Papers%20Library/Nature%20Reviews%20Molecular%20Cell%20Biology/2014/Nature%20Reviews%20Molecular%20Cell%20Biology-2014-A%20day%20in%20the%20life%20of%20the%20spliceosome-24452469_1.pdf}
}

## Conclusion
Please let me know if there are any issues with this prompt. Thanks for your help.

