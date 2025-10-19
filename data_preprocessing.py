import pandas as pd

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    # Example preprocessing
    df = df.dropna()
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data("data/raw/iris.csv", "data/processed/iris_processed.csv")