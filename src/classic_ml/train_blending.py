# ============================================================================
# BLENDING ENSEMBLE — ОБУЧЕНИЕ МОДЕЛИ
# ============================================================================
import json
import os
import pickle
import time
import numpy as np
import yaml
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score)
import lightgbm as lgb
from catboost import CatBoostClassifier


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
    """Создание базовых моделей из параметров"""
    models = []
    
    # Logistic Regression
    lr_params = params.get('blending_linreg', {})
    models.append(('LinReg', LogisticRegression(**lr_params)))
    
    # Decision Tree
    dt_params = params.get('blending_des_tree', {})
    models.append(('DesTree', DecisionTreeClassifier(**dt_params)))
    
    # Random Forest
    rf_params = params.get('blending_rf', {})
    models.append(('RF', RandomForestClassifier(**rf_params)))
    
    # Extra Trees
    et_params = params.get('blending_extra_trees', {})
    models.append(('ExtraTrees', ExtraTreesClassifier(**et_params)))
    
    # LightGBM
    lgbm_params = params.get('blending_lgbm', {})
    models.append(('LGBM', lgb.LGBMClassifier(**lgbm_params)))
    
    # Gradient Boosting
    gb_params = params.get('blending_grad_boost', {})
    models.append(('GradBoost', GradientBoostingClassifier(**gb_params)))
    
    # CatBoost
    cb_params = params.get('blending_catboost', {})
    models.append(('CatBoost', CatBoostClassifier(**cb_params)))
    
    # Naive Bayes
    nb_params = params.get('blending_nb', {})
    models.append(('NB', GaussianNB(**nb_params)))
    
    return models


def train_blending(X_train, y_train, X_test, models: list, cv_splits: int, random_state: int) -> tuple:
    """
    Обучение Blending ансамбля с OOF предсказаниями.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    # Конвертируем в numpy для позиционной индексации
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    
    oof_preds = np.zeros((len(X_train_np), len(models)))
    test_preds = np.zeros((len(X_test_np), len(models)))
    
    oof_scores = {}
    
    for i, (name, model) in enumerate(models):
        oof_fold_preds = np.zeros(len(X_train_np))
        test_fold_preds = np.zeros(len(X_test_np))
        
        for train_idx, val_idx in cv.split(X_train_np, y_train_np):
            X_train_cv, X_val_cv = X_train_np[train_idx], X_train_np[val_idx]
            y_train_cv = y_train_np[train_idx]
            
            # Копируем модель для каждого фолда
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train_cv, y_train_cv)
            
            oof_fold_preds[val_idx] = model_copy.predict_proba(X_val_cv)[:, 1]
            test_fold_preds += model_copy.predict_proba(X_test_np)[:, 1] / cv.n_splits
        
        oof_preds[:, i] = oof_fold_preds
        test_preds[:, i] = test_fold_preds
        oof_scores[name] = roc_auc_score(y_train_np, oof_fold_preds)
    
    return oof_preds, test_preds, oof_scores


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


def save_model(meta_model, output_path: str):
    """Сохранение модели в pickle"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(meta_model, f)


def save_metrics(metrics: dict, output_path: str):
    """Сохранение метрик в JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 60)
    print(" 🔄 BLENDING ENSEMBLE (С OOF)")
    print("=" * 60)

    # Пути
    data_dir = "data/processed/bank"
    params_path = "params.yaml"
    model_output = "models/blending.pkl"
    metrics_output = "metrics/blending_metrics.json"

    # Загрузка параметров
    print(f"\n📂 Загрузка параметров из: {params_path}")
    all_params = load_params(params_path)
    blending_params = all_params.get("blending", {})
    cv_splits = blending_params.get('cv_splits', 5)
    random_state = blending_params.get('random_state', 42)
    print(f"   CV splits: {cv_splits}")
    print(f"   Random state: {random_state}")

    # Загрузка данных
    print(f"\n📂 Загрузка данных из: {data_dir}")
    X_train, X_test, y_train, y_test = load_data(data_dir)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Создание базовых моделей
    print("\n🔧 Создание базовых моделей...")
    models = create_base_models(all_params)
    print(f"   Базовых моделей: {len(models)}")
    for name, _ in models:
        print(f"   - {name}")

    # Обучение Blending
    print("\n🔄 Генерация OOF предсказаний...")
    start_time = time.time()
    oof_preds, test_preds, oof_scores = train_blending(
        X_train, y_train, X_test, models, cv_splits, random_state
    )
    
    for name, score in oof_scores.items():
        print(f"   ✅ {name} — OOF ROC-AUC: {score:.4f}")

    # Обучение мета-модели
    print("\n📚 Обучение мета-модели...")
    meta_params = all_params.get('blending_meta_model', {})
    meta_model = LogisticRegression(**meta_params)
    meta_model.fit(oof_preds, y_train)

    # Финальные предсказания
    y_proba = meta_model.predict_proba(test_preds)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)
    train_time = time.time() - start_time

    # Метрики
    metrics = evaluate_model(y_test, y_pred, y_proba)
    metrics['Time (s)'] = float(train_time)

    print(f"\n⏱️ Время обучения: {train_time:.2f} сек")
    print(f"\n📊 МЕТРИКИ НА ТЕСТЕ:")
    for name, value in metrics.items():
        if name != 'Time (s)':
            print(f"  {name:12s}: {value:.4f}")
    print(f"\n🔧 Базовых моделей: {len(models)}")

    # Сохранение модели
    print(f"\n💾 Сохранение модели: {model_output}")
    save_model(meta_model, model_output)

    # Сохранение метрик
    print(f"💾 Сохранение метрик: {metrics_output}")
    save_metrics(metrics, metrics_output)

    print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)


if __name__ == "__main__":
    main()