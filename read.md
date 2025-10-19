## **Folder & File Structure with Contents**

### 1. **README.md**

```markdown
# IRIS MLOps Project

This project demonstrates an MLOps workflow for the IRIS dataset using DVC, GitHub Actions, and pytest.

## Project Structure

```

iris-mlops/
├── data/
├── src/
├── tests/
├── .github/
├── dvc.yaml
├── requirements.txt
└── README.md

````

## Setup Instructions

1. Clone the repo:
```bash
git clone https://github.com/allankisaketh/iris-mlops.git
cd iris-mlops
````

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Pull data using DVC:

```bash
dvc pull
```

4. Run tests:

```bash
pytest tests/
```

5. Run sanity check & CML report:

```bash
bash cml_report.sh
```

## Pipeline

The DVC pipeline includes:

* `data_preprocessing.py`: Cleans and prepares data
* `train_model.py`: Trains the model
* `evaluate_model.py`: Evaluates the model
* Unit tests in `tests/` validate data & evaluation

## Git Workflow

* **dev branch**: Feature development & CI testing
* **main branch**: Production-ready pipeline & CI
* Pull Requests merge `dev` → `main` after successful CI

```

---

### 2. **requirements.txt**
```

scikit-learn>=1.4.2,<1.8
pandas>=2.2.0,<3.0
joblib>=1.3.0
pytest>=8.0.0
dvc[gs]>=3.60.0,<4.0

```

---

### 3. **.gitignore**
```

**pycache**/
*.pyc
*.pyo
*.pyd
.env
.venv
*.egg-info/
dist/
build/
.dvc/tmp/

````

---

### 4. **src/__init__.py**
```python
# src package
````

---

### 5. **src/data_preprocessing.py**

```python
import pandas as pd

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    # Example preprocessing
    df = df.dropna()
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data("data/raw/iris.csv", "data/processed/iris_processed.csv")
```

---

### 6. **src/train_model.py**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(data_path, model_path):
    df = pd.read_csv(data_path)
    X = df.drop("species", axis=1)
    y = df["species"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model("data/processed/iris_processed.csv", "model.joblib")
```

---

### 7. **src/evaluate_model.py**

```python
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

def evaluate_model(data_path, model_path):
    df = pd.read_csv(data_path)
    X = df.drop("species", axis=1)
    y = df["species"]
    
    clf = joblib.load(model_path)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    return acc

if __name__ == "__main__":
    evaluate_model("data/processed/iris_processed.csv", "model.joblib")
```

---

### 8. **tests/test_data_validation.py**

```python
import pandas as pd

def test_data_not_empty():
    df = pd.read_csv("data/raw/iris.csv")
    assert not df.empty, "Dataset is empty"

def test_no_missing_values():
    df = pd.read_csv("data/raw/iris.csv")
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"
```

---

### 9. **tests/test_evaluation.py**

```python
from src.evaluate_model import evaluate_model

def test_model_accuracy():
    acc = evaluate_model("data/processed/iris_processed.csv", "model.joblib")
    assert acc > 0.7, "Model accuracy is too low"
```

---

### 10. **cml_report.sh**

```bash
#!/bin/bash

echo "Running sanity tests..."
pytest tests/ --maxfail=1 --disable-warnings -q
echo "Sanity test completed."
```

---

### 11. **.github/workflows/dev.yml**

```yaml
name: Dev CI

on:
  push:
    branches: [ dev ]
  pull_request:
    branches: [ dev ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.12
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: pytest tests/
```

---

### 12. **.github/workflows/main.yml**

```yaml
name: Main CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.12
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: pytest tests/
```

---

