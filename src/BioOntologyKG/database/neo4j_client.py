"""
Neo4j database client for BioOntologyKG
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any
import logging

from ..config.settings import settings

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        with self.driver.session(database=settings.neo4j_database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def execute_write_query(self, query: str, parameters: Dict[str, Any] = None) -> None:
        """Execute a write query"""
        with self.driver.session(database=settings.neo4j_database) as session:
            session.write_transaction(self._execute_query, query, parameters or {})
    
    @staticmethod
    def _execute_query(tx, query: str, parameters: Dict[str, Any]):
        return tx.run(query, parameters)

# Global client instance
neo4j_client = Neo4jClient()
