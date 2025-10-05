// backup neo4j database
// Neo4j Community edition database must be stopped before backup or restore
// format for backup is date based: Sep292025
neo4j-admin database dump neo4j --to-path=/Volumes/Ext2TB/neo4j_backups/MMMddyyyy

// restore neo4j database from backup
neo4j-admin database load neo4j --from-path=<backup-directory>