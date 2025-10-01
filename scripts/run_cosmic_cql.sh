#cypher-shell commands to load COSMIC data into the graph
#Edit Neo4j credentials and paths as needed
#
# Constraints for COSMIC nodes
 cypher-shell -a neo4j://localhost:7687 --format plain -u neo4j -p fjc92677 -d neo4j -f ./src/main/cql/cosmic/Cosmic_Constraints.cql
 #CosmicActionability
  cypher-shell -a neo4j://localhost:7687 --format plain -u neo4j -p fjc92677 -d neo4j -f ./src/main/cql/cosmic/cosmic_actionability.cql

# Cosmic Breakpoints
 cypher-shell -a neo4j://localhost:7687 --format plain -u neo4j -p fjc92677 -d neo4j -f ./src/main/cql/cosmic/cosmic_breakpoints.cql

#Cosmic_gene_census.cql
 cypher-shell -a neo4j://localhost:7687 --format plain -u neo4j -p fjc92677 -d neo4j -f ./src/main/cql/cosmic/cosmic_gene_census.cql
