# ============================================================================
# src/cv_decision_tree.py
# DECISION TREE + RANDOMIZED SEARCH CV — ТОЛЬКО МЕТРИКИ
# ============================================================================
import json
import os
import time
import warnings
import numpy as np
import pandas as pd
import yaml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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


def create_param_grid(cv_params: dict) -> dict:
    """Создание сетки параметров для RandomizedSearchCV"""
    custom_class_weights = [{0: 1, 1: w} for w in cv_params.get('class_weights', [2, 4, 6, 8, 10, 12])]
    custom_class_weights.extend([None, 'balanced'])
    
    param_distributions = {
        'criterion': cv_params.get('criterion', ['gini', 'entropy', 'log_loss']),
        'splitter': cv_params.get('splitter', ['best', 'random']),
        'max_depth': cv_params.get('max_depth', [None, 5, 10, 15, 20, 30]),
        'min_samples_split': cv_params.get('min_samples_split', [2, 5, 10, 20, 50]),
        'min_samples_leaf': cv_params.get('min_samples_leaf', [1, 5, 10, 20]),
        'max_features': cv_params.get('max_features', [None, 'sqrt', 'log2']),
        'class_weight': custom_class_weights
    }
    
    return param_distributions


def run_randomized_search(X_train, y_train, param_distributions: dict, cv_params: dict) -> RandomizedSearchCV:
    """Запуск RandomizedSearchCV"""
    cv_strategy = StratifiedKFold(
        n_splits=cv_params.get('cv_splits', 5), 
        shuffle=True, 
        random_state=cv_params.get('random_state', 42)
    )
    
    base_dt = DecisionTreeClassifier(random_state=cv_params.get('random_state', 42))
    
    random_search = RandomizedSearchCV(
        estimator=base_dt,
        param_distributions=param_distributions,
        n_iter=cv_params.get('n_iter', 200),
        cv=cv_strategy,
        scoring=cv_params.get('scoring', 'f1'),
        n_jobs=cv_params.get('n_jobs', -1),
        verbose=cv_params.get('verbose', 0),
        return_train_score=True,
        refit=True,
        random_state=cv_params.get('random_state', 42)
    )
    
    random_search.fit(X_train, y_train)
    
    return random_search


def extract_cv_results(random_search) -> dict:
    """Извлечение результатов CV"""
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score').head(10)
    
    def get_weight_str(cw):
        if isinstance(cw, dict): 
            return str(list(cw.values())[1])
        return str(cw)
    
    results_df['weight_ratio'] = results_df['param_class_weight'].apply(get_weight_str)
    
    best_params = random_search.best_params_
    best_score = float(random_search.best_score_)
    best_model = random_search.best_estimator_
    
    cv_results = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'scoring_metric': random_search.scoring,
        'top_10_params': results_df[['rank_test_score', 'mean_test_score', 'std_test_score', 
                                      'param_max_depth', 'param_criterion', 
                                      'param_min_samples_leaf', 'weight_ratio']].to_dict('records'),
        'tree_depth': int(best_model.get_depth()),
        'tree_leaves': int(best_model.get_n_leaves()),
        'feature_importances': best_model.feature_importances_.tolist(),
        'search_time_seconds': float(random_search.cv_results_['mean_fit_time'].mean())
    }
    
    return cv_results


def evaluate_on_test(random_search, X_test, y_test) -> dict:
    """Оценка лучшей модели на тесте"""
    best_model = random_search.best_estimator_
    
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
    print("=" * 70)
    print(" 🏆 ЛУЧШИЕ ПАРАМЕТРЫ (ПОЛНЫЙ СПИСОК)")
    print("=" * 70)
    
    print(f"\n✅ Best {cv_results['scoring_metric']} (CV): {cv_results['best_cv_score']:.6f}")
    
    print(f"\n📋 Все лучшие параметры:")
    print("-" * 50)
    for param, value in cv_results['best_params'].items():
        if param == 'class_weight' and isinstance(value, dict):
            ratio = list(value.values())[1]
            print(f"   {param:25s}: {value} ← вес миноритарного класса = {ratio}")
        else:
            print(f"   {param:25s}: {value}")
    
    print(f"\n📋 Топ-3 комбинации:")
    for i, row in enumerate(cv_results['top_10_params'][:3], 1):
        print(f"   #{i} Score: {row['mean_test_score']:.6f} ± {row['std_test_score']:.6f} | "
              f"depth={row['param_max_depth']}, criterion={row['param_criterion']}, "
              f"leaf={row['param_min_samples_leaf']}, weight={row['weight_ratio']}")
    
    print("\n" + "=" * 70)
    print(" 📊 МЕТРИКИ НА ТЕСТЕ vs CV")
    print("=" * 70)
    
    print(f"\n{'Метрика':<12} | {'CV Mean':<10} | {'Тест':<10} | {'Разница':<10}")
    print("-" * 55)
    
    cv_f1 = cv_results['best_cv_score']
    test_f1 = test_metrics['F1-Score']
    diff = test_f1 - cv_f1
    
    if diff > 0.02:
        status = "⚠️  Возможно переобучение"
    elif diff < -0.02:
        status = "✅ Недообучение"
    else:
        status = "✅ Норма"
    
    for name, value in test_metrics.items():
        if name == 'F1-Score':
            print(f"{name:<12} | {cv_f1:<10.6f} | {value:<10.6f} | {diff:+10.6f} | {status}")
        else:
            print(f"{name:<12} | {'N/A':<10} | {value:<10.6f} | {'N/A':<10}")
    
    print(f"\n🌳 Структура дерева: Глубина={cv_results['tree_depth']}, Листьев={cv_results['tree_leaves']}")
    
    print("\n" + "=" * 70)
    print(" 📊 ТОП-10 ВАЖНЫХ ПРИЗНАКОВ")
    print("=" * 70)
    
    feature_importances = cv_results['feature_importances']
    top_10_idx = np.argsort(feature_importances)[::-1][:10]
    
    for idx in top_10_idx:
        importance = feature_importances[idx]
        bar = "█" * int(importance * 40)
        print(f"   Feature_{str(idx):<16s}: {importance:.4f} {bar}")
    
    top10_sum = sum(feature_importances[idx] for idx in top_10_idx)
    print(f"\n💡 Топ-10 признаков объясняют: {top10_sum*100:.1f}% важности модели")
    
    print(f"\n⏱️  Общее время: {total_time:.2f} сек ({total_time/60:.2f} мин)")
    
    print("\n" + "=" * 70)
    print(" ✅ CV ЗАВЕРШЕН! (Модель не сохраняется, только метрики)")
    print("=" * 70)


def main():
    print("=" * 70)
    print(" 🌳 DECISION TREE + RANDOMIZED SEARCH CV")
    print("=" * 70)

    # Пути
    data_dir = "data/processed/bank"
    params_path = "params.yaml"
    metrics_output = "metrics/cv_decision_tree_metrics.json"

    # Загрузка параметров
    print(f"\n📂 Загрузка параметров из: {params_path}")
    all_params = load_params(params_path)
    cv_params = all_params.get("cv_decision_tree", {})
    
    n_iter = cv_params.get('n_iter', 200)
    cv_splits = cv_params.get('cv_splits', 5)
    scoring = cv_params.get('scoring', 'f1')
    
    print(f"   Итераций: {n_iter}")
    print(f"   CV splits: {cv_splits}")
    print(f"   Scoring: {scoring}")
    print(f"   Random state: {cv_params.get('random_state', 42)}")
    print(f"   N_jobs: {cv_params.get('n_jobs', -1)}")

    # Загрузка данных
    print(f"\n📂 Загрузка данных из: {data_dir}")
    X_train, X_test, y_train, y_test = load_data(data_dir)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Создание сетки параметров
    print("\n⚙️  Создание сетки параметров...")
    param_distributions = create_param_grid(cv_params)
    
    custom_class_weights = param_distributions['class_weight']
    print(f"   Вариантов весов: {len(custom_class_weights)}")
    print(f"   🔢 Всего комбинаций: ~{n_iter:,} (Randomized Search)")
    print(f"   🔢 С учетом {cv_splits}-Fold CV: ~{n_iter * cv_splits:,} обучений")

    # Запуск RandomizedSearchCV
    print("\n🔍 Запуск RandomizedSearchCV...")
    
    start_time = time.time()
    random_search = run_randomized_search(X_train, y_train, param_distributions, cv_params)
    total_time = time.time() - start_time
    
    print(f"\n✅ Готово за {total_time:.1f} сек ({total_time/60:.2f} мин)")

    # Извлечение результатов CV
    print("\n📊 Извлечение результатов CV...")
    cv_results = extract_cv_results(random_search)

    # Оценка на тесте
    print("\n📊 Оценка на тесте...")
    test_metrics = evaluate_on_test(random_search, X_test, y_test)

    # Печать отчёта
    print_cv_report(cv_results, test_metrics, total_time)

    # Сохранение результатов (ТОЛЬКО МЕТРИКИ, БЕЗ МОДЕЛИ)
    print(f"\n💾 Сохранение метрик: {metrics_output}")
    save_cv_results(cv_results, test_metrics, total_time, metrics_output, params_path)

    print("\n✅ CV ЗАВЕРШЕН! (Модель не сохраняется, только метрики)")
    print("=" * 70)


if __name__ == "__main__":
    main()