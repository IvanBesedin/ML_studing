import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def main():
    # Загрузка данных
    data = pd.read_csv('data/raw/bank.csv')
    
    # Препроцессинг
    data["default"] = data["default"].map({"no": 0, "yes": 1})
    data["housing"] = data["housing"].map({"no": 0, "yes": 1})
    data["loan"] = data["loan"].map({"no": 0, "yes": 1})
    data["deposit"] = data["deposit"].map({"no": 0, "yes": 1})
    
    edu_map = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
    data["education"] = data["education"].map(edu_map)
    data = pd.get_dummies(data, columns=['education'], dtype=int)
    
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    data['month_num'] = data['month'].map(month_map)
    data['month_sin'] = np.sin(2 * np.pi * data['month_num'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month_num'] / 12)
    data = data.drop(['month', 'month_num'], axis=1)
    
    data = pd.get_dummies(data, columns=['poutcome'], dtype=int)
    data = pd.get_dummies(data, columns=['contact'], dtype=int)
    data = pd.get_dummies(data, columns=['marital'], dtype=int)
    data = pd.get_dummies(data, columns=['job'], dtype=int)
    
    y = data['deposit']
    X = data.drop('deposit', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Сохранение
    os.makedirs('data/processed/bank', exist_ok=True)
    np.save('data/processed/bank/X_train.npy', X_train)
    np.save('data/processed/bank/X_test.npy', X_test)
    np.save('data/processed/bank/y_train.npy', y_train.values)
    np.save('data/processed/bank/y_test.npy', y_test.values)
    
    # Сохранение скалера
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/bank_scaler.pkl')
    
    print("✓ Данные Bank подготовлены!")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")

if __name__ == "__main__":
    main()