# ============================================================================
# STACKING ENSEMBLE — ОБУЧЕНИЕ МОДЕЛИ
# ============================================================================
import json
import os
import pickle
import time
import numpy as np
import yaml
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import StackingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score)
import lightgbm as lgb


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


def create_base_models(params: dict) -> list:
    """Создание базовых моделей для стекинга"""
    base_models = []
    
    # Logistic Regression
    lr_params = params.get('stacking_linreg', {})
    base_models.append(('LinReg', LogisticRegression(**lr_params)))
    
    # Decision Tree
    dt_params = params.get('stacking_des_tree', {})
    base_models.append(('DesTree', DecisionTreeClassifier(**dt_params)))
    
    # Random Forest
    rf_params = params.get('stacking_rf', {})
    base_models.append(('RF', RandomForestClassifier(**rf_params)))
    
    # Extra Trees
    et_params = params.get('stacking_extra_trees', {})
    base_models.append(('ExtraTrees', ExtraTreesClassifier(**et_params)))
    
    # LightGBM
    lgbm_params = params.get('stacking_lgbm', {})
    base_models.append(('LGBM', lgb.LGBMClassifier(**lgbm_params)))
    
    # Gradient Boosting
    gb_params = params.get('stacking_grad_boost', {})
    base_models.append(('GradBoost', GradientBoostingClassifier(**gb_params)))
    
    # Naive Bayes
    nb_params = params.get('stacking_nb', {})
    base_models.append(('NB', GaussianNB(**nb_params)))
    
    return base_models


def create_meta_model(params: dict) -> LogisticRegression:
    """Создание мета-модели"""
    meta_params = params.get('stacking_meta_model', {})
    return LogisticRegression(**meta_params)


def train_model(X_train, y_train, X_test, base_models: list, meta_model, 
                cv_splits: int, random_state: int, n_jobs: int, passthrough: bool) -> StackingClassifier:
    """Обучение Stacking модели"""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=cv,
        stack_method='predict_proba',
        n_jobs=n_jobs,
        passthrough=passthrough
    )
    
    stacking.fit(X_train, y_train)
    return stacking


def evaluate_model(y_test, y_pred, y_proba) -> dict:
    """Вычисление метрик модели"""
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
    print(" 📚 STACKING ENSEMBLE (С OOF)")
    print("=" * 60)

    # Пути
    data_dir = "data/processed/bank"
    params_path = "params.yaml"
    model_output = "models/stacking.pkl"
    metrics_output = "metrics/stacking_metrics.json"

    # Загрузка параметров
    print(f"\n📂 Загрузка параметров из: {params_path}")
    all_params = load_params(params_path)
    stacking_params = all_params.get("stacking", {})
    
    cv_splits = stacking_params.get('cv_splits', 5)
    random_state = stacking_params.get('random_state', 42)
    n_jobs = stacking_params.get('n_jobs', -1)
    passthrough = stacking_params.get('passthrough', False)
    
    print(f"   CV splits: {cv_splits}")
    print(f"   Random state: {random_state}")
    print(f"   N_jobs: {n_jobs}")
    print(f"   Passthrough: {passthrough}")

    # Загрузка данных
    print(f"\n📂 Загрузка данных из: {data_dir}")
    X_train, X_test, y_train, y_test = load_data(data_dir)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Создание базовых моделей
    print("\n🔧 Создание базовых моделей...")
    base_models = create_base_models(all_params)
    print(f"   Базовых моделей: {len(base_models)}")
    for name, _ in base_models:
        print(f"   - {name}")

    # Создание мета-модели
    print("\n📚 Создание мета-модели...")
    meta_model = create_meta_model(all_params)
    print(f"   Мета-модель: LogisticRegression")

    # Обучение Stacking
    print("\n🔧 Обучение Stacking модели...")
    start_time = time.time()
    model = train_model(X_train, y_train, X_test, base_models, meta_model,
                        cv_splits, random_state, n_jobs, passthrough)
    train_time = time.time() - start_time

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    metrics = evaluate_model(y_test, y_pred, y_proba)
    metrics['Time (s)'] = float(train_time)

    print(f"\n⏱️ Время обучения: {train_time:.2f} сек")
    print(f"\n📊 МЕТРИКИ НА ТЕСТЕ:")
    for name, value in metrics.items():
        if name != 'Time (s)':
            print(f"  {name:12s}: {value:.4f}")
    print(f"\n🔧 Базовых моделей: {len(base_models)}")

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