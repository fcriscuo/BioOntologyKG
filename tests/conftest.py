"""
Pytest configuration and fixtures for BioOntologyKG
"""
import pytest
import os
from pathlib import Path

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def test_data_dir():
    """Fixture providing path to test data directory"""
    return TEST_DATA_DIR

@pytest.fixture
def sample_genomic_data():
    """Fixture providing sample genomic data for testing"""
    return TEST_DATA_DIR / "sample_genomic_data.csv"

# Add more fixtures as needed for your specific testing requirements
