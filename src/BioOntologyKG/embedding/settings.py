"""
Configuration settings for BioOntologyKG
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # Embedding model
    embedding_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # Application Configuration
    log_level: str = "INFO"
    data_dir: str = "./data"
    import_batch_size: int = 1000
    
    # API Configuration
    api_host: str = "localhost"
    api_port: int = 8000
    api_debug: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
