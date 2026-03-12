import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def main():
    # Загрузка данных
    data = pd.read_csv('data/raw/adult.csv')
    
    # Препроцессинг
    data = data.drop(["education"], axis=1)
    data['sex'] = data['sex'].map({'Female': 0, 'Male': 1})
    data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})
    
    work_map = {"Without-pay": "Without-pay", "Never-worked": "Without-pay"}
    data["workclass"] = data["workclass"].map(work_map)
    
    data = pd.get_dummies(data, columns=['workclass'], dtype=int)
    data = pd.get_dummies(data, columns=['marital.status'], dtype=int)
    data = pd.get_dummies(data, columns=['occupation'], dtype=int)
    data = pd.get_dummies(data, columns=['relationship'], dtype=int)
    data = pd.get_dummies(data, columns=['race'], dtype=int)
    
    top_countries = [
        'United-States', 'Mexico', '?', 'Philippines', 'Germany',
        'Canada', 'Puerto-Rico', 'El-Salvador',
    ]
    
    def map_to_group(value, allowed_list):
        if value in allowed_list:
            return value
        else:
            return 'Other'
    
    data['native.country'] = data['native.country'].apply(
        lambda x: map_to_group(x, top_countries)
    )
    data = pd.get_dummies(data, columns=['native.country'], dtype=int)
    
    y = data['income']
    X = data.drop('income', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Сохранение
    os.makedirs('data/processed/adult', exist_ok=True)
    np.save('data/processed/adult/X_train.npy', X_train)
    np.save('data/processed/adult/X_test.npy', X_test)
    np.save('data/processed/adult/y_train.npy', y_train.values)
    np.save('data/processed/adult/y_test.npy', y_test.values)
    
    # Сохранение скалера
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/adult_scaler.pkl')
    
    print("✓ Данные Adult подготовлены!")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")

if __name__ == "__main__":
    main()