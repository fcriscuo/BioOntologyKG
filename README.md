# BioOntologyKG

A genomic data import platform built with Python3 and Neo4j.

## Features

- Import genomic data from delimited text files
- RESTful API integration for data import
- Neo4j graph database for genomic data storage
- Cypher Query Language (CQL) scripts for data operations
- Comprehensive testing suite

## Setup

1. Install uv package manager
2. Install dependencies:
   ```bash
   uv sync
   ```

3. Start Neo4j database:
   ```bash
   docker-compose up -d
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Usage

### Running Import Scripts

```bash
uv run scripts/import_genomic_data.py
```

### Running Tests

```bash
uv run pytest
```

## Project Structure

- `src/` - Main source code
- `scripts/` - Standalone scripts
- `cql/` - Cypher Query Language scripts
- `tests/` - Test suite
- `prompts/` - LLM prompts
- `docs/` - Documentation
- `data/` - Data files
- `logs/` - Application logs

## Development

See `docs/developer_guide/` for development instructions.
