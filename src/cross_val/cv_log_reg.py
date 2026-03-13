# ============================================================================
# ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ + GRID SEARCH CV — ТОЛЬКО МЕТРИКИ
# ============================================================================
import json
import os
import time
import warnings
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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


def create_param_grid(cv_params: dict) -> list:
    """Создание сетки параметров для GridSearchCV"""
    custom_class_weights = [{0: 1, 1: w} for w in cv_params.get('class_weights', [2, 4, 6, 8, 10, 12])]
    custom_class_weights.extend([None, 'balanced'])
    
    param_grid = [
        {
            'solver': ['lbfgs'],
            'penalty': ['l2', 'none'],
            'C': cv_params.get('C_values', [0.001, 0.01, 0.1, 1, 10, 100]),
            'class_weight': custom_class_weights,
            'max_iter': cv_params.get('max_iter_values', [1000, 5000]),
            'tol': [1e-4]
        },
        {
            'solver': ['liblinear'],
            'penalty': ['l1', 'l2'],
            'C': cv_params.get('C_values', [0.001, 0.01, 0.1, 1, 10, 100]),
            'class_weight': custom_class_weights,
            'max_iter': cv_params.get('max_iter_values', [1000, 5000]),
            'tol': [1e-4]
        },
        {
            'solver': ['saga'],
            'penalty': ['elasticnet'],
            'C': cv_params.get('C_values', [0.001, 0.01, 0.1, 1, 10, 100]),
            'l1_ratio': cv_params.get('l1_ratio_values', [0.15, 0.5, 0.85]),
            'class_weight': custom_class_weights,
            'max_iter': cv_params.get('max_iter_values', [1000, 5000]),
            'tol': [1e-4]
        },
    ]
    
    return param_grid, custom_class_weights


def run_grid_search(X_train, y_train, param_grid: list, cv_params: dict) -> GridSearchCV:
    """Запуск GridSearchCV"""
    cv_strategy = StratifiedKFold(
        n_splits=cv_params.get('cv_splits', 5), 
        shuffle=True, 
        random_state=cv_params.get('random_state', 42)
    )
    
    base_log_reg = LogisticRegression(random_state=cv_params.get('random_state', 42))
    
    grid_search = GridSearchCV(
        estimator=base_log_reg,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=cv_params.get('n_jobs', -1),
        verbose=cv_params.get('verbose', 0),
        return_train_score=True,
        refit=True
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search


def extract_cv_results(grid_search) -> dict:
    """Извлечение результатов CV"""
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score').head(10)
    
    def extract_weight_ratio(row):
        cw = row.get('param_class_weight')
        if isinstance(cw, dict):
            return list(cw.values())[1]
        elif cw == 'balanced':
            return 'balanced'
        else:
            return 'None'
    
    results_df['weight_ratio'] = results_df.apply(extract_weight_ratio, axis=1)
    
    best_params = grid_search.best_params_
    best_score = float(grid_search.best_score_)
    
    cv_results = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'top_10_params': results_df[['rank_test_score', 'mean_test_score', 'std_test_score', 
                                      'param_solver', 'param_penalty', 'param_C', 
                                      'weight_ratio', 'param_max_iter']].to_dict('records'),
        'cv_time_seconds': float(grid_search.cv_results_['mean_fit_time'].mean())
    }
    
    return cv_results


def evaluate_on_test(grid_search, X_test, y_test) -> dict:
    """Оценка лучшей модели на тесте"""
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
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


def print_cv_report(cv_results: dict, test_metrics: dict, total_time: float):
    """Печать отчёта о CV"""
    print("=" * 90)
    print(" 🏆 ЛУЧШИЕ ПАРАМЕТРЫ")
    print("=" * 90)
    
    print(f"\n✅ Лучшие параметры:")
    for param, value in cv_results['best_params'].items():
        if param == 'class_weight' and isinstance(value, dict):
            ratio = list(value.values())[1]
            print(f"   {param:20s}: {value}  ← вес миноритарного класса = {ratio}")
        else:
            print(f"   {param:20s}: {value}")
    
    print(f"\n📈 Лучший ROC-AUC (CV): {cv_results['best_cv_score']:.6f}")
    
    print("\n" + "=" * 90)
    print(" 📋 ТОП-10 КОМБИНАЦИЙ ПАРАМЕТРОВ")
    print("=" * 90)
    
    for i, row in enumerate(cv_results['top_10_params'][:10], 1):
        print(f"   #{i} ROC-AUC: {row['mean_test_score']:.6f} ± {row['std_test_score']:.6f} | "
              f"solver={row['param_solver']}, C={row['param_C']}, weight={row['weight_ratio']}")
    
    print("\n" + "=" * 90)
    print(" 📊 ФИНАЛЬНЫЕ МЕТРИКИ НА ТЕСТЕ")
    print("=" * 90)
    
    print(f"\n📈 МЕТРИКИ:")
    print("-" * 50)
    for name, value in test_metrics.items():
        print(f"  {name:15s}: {value:.6f}")
    
    print("\n" + "=" * 90)
    print(" 🔍 СРАВНЕНИЕ: CV (Mean) vs ТЕСТ")
    print("=" * 90)
    
    cv_auc = cv_results['best_cv_score']
    test_auc = test_metrics['ROC-AUC']
    diff = test_auc - cv_auc
    
    if diff > 0.02:
        status = "⚠️  Возможно переобучение"
    elif diff < -0.02:
        status = "✅ Недообучение"
    else:
        status = "✅ Норма"
    
    print(f"\n{'Метрика':<15s} | {'CV Mean':<12s} | {'Тест':<12s} | {'Разница':<12s} | {'Статус':<10s}")
    print("-" * 70)
    print(f"{'ROC-AUC':<15s} | {cv_auc:<12.6f} | {test_auc:<12.6f} | {diff:+12.6f} | {status:<10s}")
    
    print(f"\n⏱️  Общее время: {total_time:.2f} сек ({total_time/60:.2f} мин)")
    
    print("\n" + "=" * 90)
    print(" ✅ GRID SEARCH CV ЗАВЕРШЕН!")
    print("=" * 90)


def main():
    print("=" * 90)
    print(" ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ + GRID SEARCH CV")
    print("=" * 90)

    # Пути
    data_dir = "data/processed/bank"
    params_path = "params.yaml"
    metrics_output = "metrics/cv_log_reg_metrics.json"

    # Загрузка параметров
    print(f"\n📂 Загрузка параметров из: {params_path}")
    all_params = load_params(params_path)
    cv_params = all_params.get("cv_log_reg", {})
    print(f"   CV splits: {cv_params.get('cv_splits', 5)}")
    print(f"   Random state: {cv_params.get('random_state', 42)}")
    print(f"   N_jobs: {cv_params.get('n_jobs', -1)}")

    # Загрузка данных
    print(f"\n📂 Загрузка данных из: {data_dir}")
    X_train, X_test, y_train, y_test = load_data(data_dir)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Создание сетки параметров
    print("\n⚙️  Создание сетки параметров...")
    param_grid, class_weights = create_param_grid(cv_params)
    
    total_combinations = 0
    for grid in param_grid:
        combos = 1
        for key, values in grid.items():
            combos *= len(values)
        total_combinations += combos
    
    print(f"   🔢 Всего комбинаций: ~{total_combinations:,}")
    print(f"   🔢 С учетом {cv_params.get('cv_splits', 5)}-Fold CV: ~{total_combinations * cv_params.get('cv_splits', 5):,} обучений")

    # Запуск GridSearchCV
    print("\n🔍 Запуск GridSearchCV...")
    print("-" * 90)
    
    start_time = time.time()
    grid_search = run_grid_search(X_train, y_train, param_grid, cv_params)
    total_time = time.time() - start_time
    
    print("-" * 90)
    print(f"\n⏱️  Фактическое время выполнения: {total_time:.3f} сек ({total_time/60:.2f} мин)")

    # Извлечение результатов CV
    print("\n📊 Извлечение результатов CV...")
    cv_results = extract_cv_results(grid_search)

    # Оценка на тесте
    print("\n📊 Оценка на тесте...")
    test_metrics = evaluate_on_test(grid_search, X_test, y_test)

    # Печать отчёта
    print_cv_report(cv_results, test_metrics, total_time)

    # Сохранение результатов (ТОЛЬКО МЕТРИКИ, БЕЗ МОДЕЛИ)
    print(f"\n💾 Сохранение метрик: {metrics_output}")
    save_cv_results(cv_results, test_metrics, total_time, metrics_output, params_path)

    print("\n✅ CV ЗАВЕРШЕН! (Модель не сохраняется, только метрики)")
    print("=" * 90)


if __name__ == "__main__":
    main()