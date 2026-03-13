# ============================================================================
# CATBOOST + CUSTOM RANDOMIZED SEARCH CV — ТОЛЬКО МЕТРИКИ
# ============================================================================
import json
import os
import time
import warnings
import random
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score)

warnings.filterwarnings('ignore')


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


def generate_random_params(param_dist: dict, n_iter: int, random_state: int = 42) -> list:
    """Генерация случайных комбинаций параметров"""
    random.seed(random_state)
    params_list = []
    keys = list(param_dist.keys())
    
    for _ in range(n_iter):
        param_set = {key: random.choice(param_dist[key]) for key in keys}
        params_list.append(param_set)
    
    return params_list


def run_custom_cv(X_train, y_train, random_params: list, cv_splits: int, 
                  random_state: int) -> tuple:
    """Запуск кастомной кросс-валидации для CatBoost"""
    cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    
    # Преобразуем в numpy для гарантированной позиционной индексации
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    
    # 🔧 Гарантируем что это numpy array
    X_train_np = np.array(X_train_np)
    y_train_np = np.array(y_train_np)
    
    results = []
    
    for idx, params in enumerate(random_params):
        cv_scores = []
        
        for train_idx, val_idx in cv_strategy.split(X_train_np, y_train_np):
            X_train_cv = X_train_np[train_idx]
            X_val_cv = X_train_np[val_idx]
            y_train_cv = np.array(y_train_np[train_idx])  # 🔧 Исправлено
            y_val_cv = np.array(y_train_np[val_idx])      # 🔧 Исправлено
            
            model = CatBoostClassifier(
                **params,
                random_state=random_state,
                verbose=0,
                allow_writing_files=False
            )
            
            model.fit(X_train_cv, y_train_cv)
            y_proba = model.predict_proba(X_val_cv)[:, 1]
            score = roc_auc_score(y_val_cv, y_proba)
            cv_scores.append(score)
        
        results.append({
            'params': params,
            'mean_test_score': np.mean(cv_scores),
            'std_test_score': np.std(cv_scores),
            'cv_scores': cv_scores
        })
        
        if (idx + 1) % 10 == 0:
            print(f"   ▶️  {idx + 1}/{len(random_params)} завершено")
    
    return results, X_train_np.shape


def extract_cv_results(results: list, param_distributions: dict) -> dict:
    """Извлечение результатов CV"""
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mean_test_score', ascending=False).reset_index(drop=True)
    
    for key in param_distributions.keys():
        results_df[f'param_{key}'] = results_df['params'].apply(lambda x: x.get(key))
    
    results_df['rank_test_score'] = results_df['mean_test_score'].rank(ascending=False).astype(int)
    
    best_params = results_df.iloc[0]['params']
    best_score = float(results_df.iloc[0]['mean_test_score'])
    
    cv_results = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'top_3_params': results_df[['rank_test_score', 'mean_test_score', 'std_test_score', 
                                     'param_iterations', 'param_depth', 
                                     'param_learning_rate', 'param_scale_pos_weight']].head(3).to_dict('records'),
        'depth_stats': results_df.groupby('param_depth')['mean_test_score'].agg(['mean', 'std', 'count']).round(4).to_dict(),
        'lr_stats': results_df.groupby('param_learning_rate')['mean_test_score'].agg(['mean', 'std', 'count']).round(4).to_dict(),
        'scale_pos_weight_stats': results_df.groupby('param_scale_pos_weight')['mean_test_score'].agg(['mean', 'std', 'count']).round(4).to_dict(),
        'iterations_stats': results_df.groupby('param_iterations')['mean_test_score'].agg(['mean', 'std', 'count']).round(4).to_dict()
    }
    
    return cv_results, results_df


def train_final_model(X_train, y_train, best_params: dict, random_state: int) -> CatBoostClassifier:
    """Обучение финальной модели на всех данных"""
    best_cb = CatBoostClassifier(
        **best_params,
        random_state=random_state,
        verbose=0,
        allow_writing_files=False
    )
    
    best_cb.fit(X_train, y_train)
    return best_cb


def evaluate_on_test(model, X_test, y_test) -> dict:
    """Оценка модели на тесте"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': float(accuracy_score(y_test, y_pred)),
        'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'F1-Score': float(f1_score(y_test, y_pred, zero_division=0)),
        'ROC-AUC': float(roc_auc_score(y_test, y_proba)),
        'PR-AUC': float(average_precision_score(y_test, y_proba))
    }
    
    return metrics


def save_cv_results(cv_results: dict, test_metrics: dict, total_time: float, 
                    output_path: str, params_path: str):
    """Сохранение результатов CV в JSON"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = {
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'total_time_seconds': total_time,
        'params_file': params_path
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def print_cv_report(cv_results: dict, test_metrics: dict, total_time: float, 
                    results_df: pd.DataFrame, data_shape: tuple):
    """Печать отчёта о CV"""
    print("\n" + "=" * 70)
    print(" 🏆 ЛУЧШИЕ ПАРАМЕТРЫ (ПОЛНЫЙ СПИСОК)")
    print("=" * 70)
    
    print(f"\n✅ Best ROC-AUC (CV): {cv_results['best_cv_score']:.6f}")
    
    print(f"\n📋 Все лучшие параметры (из поиска):")
    print("-" * 50)
    for param, value in cv_results['best_params'].items():
        if param == 'scale_pos_weight' and value is not None:
            print(f"   {param:25s}: {value} ← вес миноритарного класса")
        elif param == 'scale_pos_weight' and value is None:
            print(f"   {param:25s}: {value} ← без взвешивания")
        else:
            print(f"   {param:25s}: {value}")
    
    print(f"\n📋 Топ-3 комбинации:")
    print(results_df[['rank_test_score', 'mean_test_score', 'std_test_score', 
                      'param_iterations', 'param_depth', 'param_learning_rate', 
                      'param_scale_pos_weight']].head(3).to_string(index=False))
    
    print("\n" + "=" * 70)
    print(" 📊 МЕТРИКИ НА ТЕСТЕ vs CV")
    print("=" * 70)
    
    print(f"\n⏱️  Время обучения: {total_time:.1f} сек")
    
    print(f"\n{'Метрика':<12} | {'CV Mean':<10} | {'Тест':<10} | {'Разница':<10}")
    print("-" * 55)
    
    cv_auc = cv_results['best_cv_score']
    test_auc = test_metrics['ROC-AUC']
    diff = test_auc - cv_auc
    
    if diff > 0.02:
        status = "⚠️  Возможно переобучение"
    elif diff < -0.02:
        status = "✅ Недообучение"
    else:
        status = "✅ Норма"
    
    for name, value in test_metrics.items():
        if name == 'ROC-AUC':
            print(f"{name:<12} | {cv_auc:<10.6f} | {value:<10.6f} | {diff:+10.6f} | {status}")
        else:
            print(f"{name:<12} | {'N/A':<10} | {value:<10.6f} | {'N/A':<10}")
    
    print("\n" + "=" * 70)
    print(" 📊 АНАЛИЗ ПО КЛЮЧЕВЫМ ПАРАМЕТРАМ")
    print("=" * 70)
    
    print(f"\n📈 По глубине деревьев:")
    depth_stats = results_df.groupby('param_depth')['mean_test_score'].agg(['mean', 'std', 'count']).round(4)
    print(depth_stats.to_string())
    
    print(f"\n📈 По скорости обучения (LR):")
    lr_stats = results_df.groupby('param_learning_rate')['mean_test_score'].agg(['mean', 'std', 'count']).round(4)
    print(lr_stats.to_string())
    
    print(f"\n📈 По scale_pos_weight (баланс классов):")
    spw_stats = results_df.groupby('param_scale_pos_weight')['mean_test_score'].agg(['mean', 'std', 'count']).round(4)
    print(spw_stats.to_string())
    
    print(f"\n📈 По количеству итераций:")
    iter_stats = results_df.groupby('param_iterations')['mean_test_score'].agg(['mean', 'std', 'count']).round(4)
    print(iter_stats.to_string())
    
    print(f"\n⏱️  Общее время: {total_time:.2f} сек ({total_time/60:.2f} мин)")
    
    print("\n" + "=" * 70)
    print(" ✅ CV ЗАВЕРШЕН! (Модель не сохраняется, только метрики)")
    print("=" * 70)


def main():
    print("=" * 70)
    print(" 🔴 CATBOOST + CUSTOM RANDOMIZED SEARCH CV")
    print("=" * 70)

    # Пути
    data_dir = "data/processed/bank"
    params_path = "params.yaml"
    metrics_output = "metrics/cv_catboost_metrics.json"

    # Загрузка параметров
    print(f"\n📂 Загрузка параметров из: {params_path}")
    all_params = load_params(params_path)
    cv_params = all_params.get("cv_catboost", {})
    
    n_iter = cv_params.get('n_iter', 50)
    cv_splits = cv_params.get('cv_splits', 5)
    random_state = cv_params.get('random_state', 42)
    
    param_distributions = {
        'iterations': cv_params.get('iterations', [100, 200, 300, 500]),
        'depth': cv_params.get('depth', [4, 6, 8, 10]),
        'learning_rate': cv_params.get('learning_rate', [0.01, 0.03, 0.05, 0.1, 0.2]),
        'l2_leaf_reg': cv_params.get('l2_leaf_reg', [1, 3, 5, 10, 30]),
        'subsample': cv_params.get('subsample', [0.7, 0.8, 0.9, 1.0]),
        'colsample_bylevel': cv_params.get('colsample_bylevel', [0.7, 0.8, 0.9, 1.0]),
        'scale_pos_weight': cv_params.get('scale_pos_weight', [1, 2, 5, 10, None]),
        'min_data_in_leaf': cv_params.get('min_data_in_leaf', [5, 10, 20, 50]),
    }
    
    print(f"   Итераций: {n_iter}")
    print(f"   CV splits: {cv_splits}")
    print(f"   Random state: {random_state}")
    print(f"   ⏱️  Ожидаемое время: ~{(n_iter * cv_splits * 0.5) / 60:.1f} мин")

    # Загрузка данных
    print(f"\n📂 Загрузка данных из: {data_dir}")
    X_train, X_test, y_train, y_test = load_data(data_dir)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Генерация случайных параметров
    print("\n🔍 Генерация параметров...")
    random_params = generate_random_params(param_distributions, n_iter, random_state)

    # Запуск кастомной CV
    print("\n🔍 Запуск кросс-валидации...")
    start_time = time.time()
    results, data_shape = run_custom_cv(X_train, y_train, random_params, cv_splits, random_state)
    total_time = time.time() - start_time
    
    print(f"\n✅ Готово за {total_time:.1f} сек ({total_time/60:.2f} мин)")

    # Извлечение результатов
    print("\n📊 Извлечение результатов CV...")
    cv_results, results_df = extract_cv_results(results, param_distributions)

    # Обучение финальной модели (для оценки на тесте)
    print("\n🔧 Обучение финальной модели на всех данных...")
    best_model = train_final_model(X_train, y_train, cv_results['best_params'], random_state)

    # Оценка на тесте
    print("\n📊 Оценка на тесте...")
    test_metrics = evaluate_on_test(best_model, X_test, y_test)

    # Печать отчёта
    print_cv_report(cv_results, test_metrics, total_time, results_df, data_shape)

    # Сохранение результатов (ТОЛЬКО МЕТРИКИ, БЕЗ МОДЕЛИ)
    print(f"\n💾 Сохранение метрик: {metrics_output}")
    save_cv_results(cv_results, test_metrics, total_time, metrics_output, params_path)

    print("\n✅ CV ЗАВЕРШЕН! (Модель не сохраняется, только метрики)")
    print("=" * 70)


if __name__ == "__main__":
    main()