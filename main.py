from scrape import scrape
from analysis import make_graphs
from train import create_model

if __name__ == "__main__":
    print("Starting data scraping...")
    scrape()  # Scrape data and save to SQLite

    print("\nStarting data analysis and visualization...")
    make_graphs()  # Generate plots and save to assets/

    print("\nStarting model training...")
    create_model()  # Train models and save artifact
