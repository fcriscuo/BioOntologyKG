# run the following commands to backup the system and neo4j databases
cd $NEO4J_HOME
./bin/neo4j-admin database dump --expand-commands system --to-path=./backups &&
neo4j-admin database dump --expand-commands neo4j --to-path=./backups