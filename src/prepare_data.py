import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # <-- Добавили это
import os

def main():
    # Загрузка данных
    data = pd.read_csv('data/raw/adult.csv')
    
    # Препроцессинг (код из вашего ноутбука)
    data = data.drop(["education"], axis=1)
    data['sex'] = data['sex'].map({'Female': 0, 'Male': 1})
    data['income'] = data['income'].map({'<=50K': 0, '>50K': 1})
    
    # One-hot encoding
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
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Создаем папку для результатов, если нет
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Сохранение обработанных данных
    train_df = pd.DataFrame(X_train)
    train_df['target'] = y_train.values
    train_df.to_csv('data/processed/train.csv', index=False)
    
    test_df = pd.DataFrame(X_test)
    test_df['target'] = y_test.values
    test_df.to_csv('data/processed/test.csv', index=False)
    
    # Сохранение скалера для инференса
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("✓ Данные подготовлены и сохранены в data/processed/")

if __name__ == "__main__":
    main()