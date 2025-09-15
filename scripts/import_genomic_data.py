#!/usr/bin/env python3
"""
Main script for importing genomic data into Neo4j
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from BioOntologyKG.database.neo4j_client import neo4j_client

def main():
    parser = argparse.ArgumentParser(description="Import genomic data into Neo4j")
    parser.add_argument("--file", "-f", required=True, help="Path to genomic data file")
    parser.add_argument("--type", "-t", choices=["genes", "variants", "samples"], 
                       required=True, help="Type of data to import")
    
    args = parser.parse_args()
    
    print(f"Importing {args.type} from {args.file}")
    
    # TODO: Implement actual import logic
    print("Import completed successfully!")

if __name__ == "__main__":
    main()
