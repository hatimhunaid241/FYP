"""
Project Setup Script
Run this first to create all necessary directories
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create all project directories"""
    
    base_dir = Path(__file__).parent
    
    directories = [
        # Config
        "config",
        
        # Data directories
        "data/raw/products",
        "data/raw/reviews",
        "data/processed",
        "data/embeddings",
        "data/database",
        
        # Source code
        "src/data_collection",
        "src/preprocessing",
        "src/embeddings",
        "src/clustering",
        "src/sentiment",
        "src/search",
        "src/evaluation",
        "src/utils",
        
        # Notebooks
        "notebooks",
        
        # Experiments
        "experiments/clustering",
        "experiments/embeddings",
        
        # Results
        "results/figures",
        "results/tables",
        "results/clusters",
        
        # Tests
        "tests",
        
        # Docs
        "docs",
        
        # Logs
        "logs",
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
        
        # Create __init__.py for Python packages
        if directory.startswith("src/"):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()

if __name__ == "__main__":
    print("Setting up project structure...")
    create_directory_structure()
    print("\n✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Configure API settings in config/config.yaml")
    print("3. Run data collection: python src/data_collection/collect_products.py")
