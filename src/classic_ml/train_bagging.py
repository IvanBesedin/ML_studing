# ============================================================================
# BAGGING ENSEMBLE — ОБУЧЕНИЕ МОДЕЛИ
# ============================================================================
import json
import os
import pickle
import time
import numpy as np
import yaml
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score)


def load_params(params_path: str) -> dict:
    """Загрузка параметров из YAML файла"""
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(data_dir: str) -> tuple:
    """Загрузка данных из .npy файлов"""
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_train, X_test, y_train, y_test


def create_base_estimator(base_params: dict) -> RandomForestClassifier:
    """Создание базового эстиматора (RandomForest)"""
    base_params = base_params.copy()
    random_state = base_params.pop('random_state', 42)
    
    rf = RandomForestClassifier(random_state=random_state, **base_params)
    return rf


def train_model(X_train, y_train, bagging_params: dict, base_params: dict) -> BaggingClassifier:
    """
    Обучение Bagging модели.
    """
    # Создаём базовый эстиматор
    base_estimator = create_base_estimator(base_params)
    
    # Параметры BaggingClassifier
    bagging_kwargs = bagging_params.copy()
    random_state = bagging_kwargs.pop('random_state', 42)
    
    bagging = BaggingClassifier(
        estimator=base_estimator,
        random_state=random_state,
        **bagging_kwargs
    )
    
    bagging.fit(X_train, y_train)
    return bagging


def evaluate_model(model, X_test, y_test) -> dict:
    """Вычисление метрик модели"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': float(accuracy_score(y_test, y_pred)),
        'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'F1-Score': float(f1_score(y_test, y_pred, zero_division=0)),
        'ROC-AUC': float(roc_auc_score(y_test, y_proba)),
        'PR-AUC': float(average_precision_score(y_test, y_proba)),
    }
    return metrics


def save_model(model, output_path: str):
    """Сохранение модели в pickle"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)


def save_metrics(metrics: dict, output_path: str):
    """Сохранение метрик в JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 60)
    print(" 🎒 BAGGING ENSEMBLE")
    print("=" * 60)

    # Пути
    data_dir = "data/processed/bank"
    params_path = "params.yaml"
    model_output = "models/bagging.pkl"
    metrics_output = "metrics/bagging_metrics.json"

    # Загрузка параметров
    print(f"\n📂 Загрузка параметров из: {params_path}")
    all_params = load_params(params_path)
    bagging_params = all_params.get("bagging", {})
    base_params = all_params.get("bagging_base_estimator", {})
    print(f"   Параметры Bagging: {bagging_params}")
    print(f"   Параметры базового эстиматора: {base_params}")

    # Загрузка данных
    print(f"\n📂 Загрузка данных из: {data_dir}")
    X_train, X_test, y_train, y_test = load_data(data_dir)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Обучение модели
    print("\n🔧 Обучение модели...")
    start_time = time.time()
    model = train_model(X_train, y_train, bagging_params, base_params)
    train_time = time.time() - start_time

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    metrics = {
        'Accuracy': float(accuracy_score(y_test, y_pred)),
        'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'F1-Score': float(f1_score(y_test, y_pred, zero_division=0)),
        'ROC-AUC': float(roc_auc_score(y_test, y_proba)),
        'PR-AUC': float(average_precision_score(y_test, y_proba)),
        'Time (s)': float(train_time)
    }

    print(f"\n⏱️ Время обучения: {train_time:.2f} сек")
    print(f"\n📊 МЕТРИКИ НА ТЕСТЕ:")
    for name, value in metrics.items():
        if name != 'Time (s)':
            print(f"  {name:12s}: {value:.4f}")
    print(f"\n🌳 Деревьев: {len(model.estimators_)}")

    # Сохранение модели
    print(f"\n💾 Сохранение модели: {model_output}")
    save_model(model, model_output)

    # Сохранение метрик
    print(f"💾 Сохранение метрик: {metrics_output}")
    save_metrics(metrics, metrics_output)

    print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)


if __name__ == "__main__":
    main()