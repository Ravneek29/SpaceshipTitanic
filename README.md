# Spaceship Titanic Survival Prediction

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Predicted passenger survival on the Spaceship Titanic using structured tabular data with mixed feature types (categorical, boolean, numerical). Implemented and compared 7 classical ML models plus a neural network.

---

## Dataset

- **Train set:** 8,693 passengers (after cleaning: 8,069)
- **Test set:** 4,277 passengers (after cleaning: 3,990)
- **Features:** PassengerId, HomePlanet, CryoSleep, Cabin, Destination, Age, VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck

---

## Installation
To set up this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/spaceship-titanic.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd spaceship-titanic
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To use this project, open the Jupyter Notebook `SpaceshipTitanic.ipynb` in your preferred environment. Follow the steps outlined in the notebook to preprocess the data, train the model, and make predictions.

## Project Structure
- `SpaceshipTitanic.ipynb`: The main Jupyter Notebook containing the code and analysis.
- `data/`: Directory containing the dataset.
- `models/`: Directory to save trained models.
- `requirements.txt`: File containing the list of dependencies.


## Results

| Model | Best CV Accuracy | Best Parameters |
|---|---|---|
| **Random Forest** ⭐ | **79.12%** | max_depth=10, n_estimators=200, min_samples_leaf=4 |
| SVM (RBF) | 79.04% | C=10, gamma=auto, kernel=rbf |
| Logistic Regression | 77.93% | C=0.01, solver=newton-cg |
| AdaBoost | 77.63% | learning_rate=1, n_estimators=100 |
| Ridge Classifier | 77.59% | alpha=1000 |
| KNN | 76.90% | metric=euclidean, n_neighbors=7 |
| Decision Tree | 76.25% | max_depth=10, min_samples_leaf=4 |
| Neural Network | **78.25% val accuracy** | Dense(64→32→1), Dropout=0.5, EarlyStopping |

All classical models tuned with **5-fold cross-validation GridSearchCV**.

---


## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your improvements or bug fixes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Methodology

**Data Cleaning & Feature Engineering**
- Imputed missing values: median for Age, mode for HomePlanet
- Removed outliers using IQR method (2.5x) on Age and total_spend
- Engineered `total_spend` feature (sum of all spending columns)
- Split Cabin into CabinDeck, CabinNum, CabinSide

**Preprocessing Pipeline**
- Built sklearn `Pipeline` + `ColumnTransformer` for reproducible preprocessing
- `OneHotEncoder` for categorical features (HomePlanet, Destination, CabinDeck, CabinSide)
- `StandardScaler` + `SimpleImputer` for numerical features
- Binary encoder for CryoSleep and VIP

**Model Training**
- GridSearchCV with 5-fold cross-validation on all models
- Neural Network: Dense(64, relu) → Dropout(0.5) → Dense(32, relu) → Dropout(0.5) → Dense(1, sigmoid), trained with EarlyStopping (patience=5)

---

## What I Learned

Building this project taught me how to design reusable sklearn Pipelines that prevent data leakage between train and test sets — something that's easy to get wrong with manual preprocessing. Comparing 7 models side-by-side showed me that ensemble methods (Random Forest, AdaBoost) and kernel-based methods (SVM) consistently outperformed simpler linear models on this dataset, likely because of the non-linear relationships between spending features and survival. The neural network matched the classical models closely despite the relatively small dataset size, which I found surprising.

---

## Tech Stack

Python · TensorFlow/Keras · scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Kaggle

---

## How to Run

```bash
git clone https://github.com/Ravneek29/SpaceshipTitanic.git
cd SpaceshipTitanic
pip install -r requirements.txt
# Open SpaceshipTitanic.ipynb in Jupyter or VS Code
```

Download the dataset from [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic/data) and place `train.csv` and `test.csv` in the project directory.
