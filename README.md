# League-of-Legends-Champion-Classifier-using-K-Nearest-Neighbors-KNN-
This project leverages machine learning to classify champions in the popular game League of Legends based on their attributes. The model predicts a champion's class (e.g., Assassin, Tank, Mage) using features such as Base HP, Base Mana, Difficulty, Role, and Range Type

![image](https://github.com/user-attachments/assets/af6df1b0-9c84-43d0-853a-eba2899465c2)


# League of Legends Champion Classifier using K-Nearest Neighbors (KNN)

This project leverages machine learning to classify champions in the popular game **League of Legends** based on their attributes. The model predicts a champion's class (e.g., Assassin, Tank, Mage) using features such as Base HP, Base Mana, Difficulty, Role, and Range Type. This project showcases a complete ML pipeline, including data preprocessing, feature engineering, and model evaluation.

---

## Objective

To predict the **class** of a League of Legends champion based on their attributes using a **K-Nearest Neighbors (KNN)** model.

---

## Dataset

The dataset contains attributes of League of Legends champions, including:

- **Base HP**: Champion's health at the start of the game.
- **Base Mana**: Champion's mana at the start of the game.
- **Difficulty**: Level of complexity to play the champion (e.g., Beginner, Intermediate, Advanced).
- **Role**: Primary role of the champion (e.g., Top, Mid, Jungle, Support, ADC).
- **Range type**: Whether the champion attacks at range or melee.
- **Class**: The target variable, representing the champion's class (e.g., Assassin, Mage, Tank).

---

## Methodology

### 1. **Data Preprocessing**

- **Encoding Categorical Data**: Categorical features (`Difficulty`, `Role`, `Range type`) were converted to numerical values using `LabelEncoder`.
- **Feature Scaling**: Numerical features (`Base HP`, `Base Mana`, `Difficulty`) were normalized using `StandardScaler` to ensure uniformity and improve model performance.

### 2. **Model Training**

- Split the data into training and testing sets using an 80-20 ratio.
- Trained a KNN classifier with `k=4` to predict the champion class.

### 3. **Prediction Pipeline**

- A user-friendly prediction pipeline was built to accept champion attributes, process the input, and return the predicted class as output.

---
### Acknowledgments

This project uses the **League of Legends Dataset** created by [Gabriel](https://www.kaggle.com/gabkgonzales) and [Marwan](https://www.kaggle.com/marwant1), available on [Kaggle](https://www.kaggle.com/datasets/gabkgonzales/league-of-legends-dataset). 

A big thank you to them for providing this excellent dataset for the community to explore and analyze.

## Implementation

### Data Preprocessing

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Encode the categorical variables
le = LabelEncoder()
df['Difficulty'] = le.fit_transform(df['Difficulty'])
df['Role'] = le.fit_transform(df['Role'])
df['Range type'] = le.fit_transform(df['Range type'])
df['Classes'] = le.fit_transform(df['Classes'])

#Normalize the numerical features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(df[['Base HP', 'Base Mana', 'Difficulty']])
