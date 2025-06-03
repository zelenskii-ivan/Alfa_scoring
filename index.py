import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from pathlib import Path
import gc
import logging
import sys
import time
import joblib
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler

CHUNK_SIZE = 100_000

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=10,
    num_leaves=80,
    min_child_samples=30,
    subsample=0.95,
    colsample_bytree=0.95,
    random_state=42,
    n_jobs=4,
    class_weight='balanced',
    boosting_type='gbdt',
    reg_alpha=0.2,
    reg_lambda=0.7,
    verbose=-1
)

def create_features(df: pd.DataFrame, numeric_cols: List[str], precomputed_stats: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    try:
        result = df.copy()
        # Базовые статистики по группам
        if precomputed_stats is not None:
            new_cols = {}
            for col in numeric_cols:
                mean_key = f'{col}_mean'
                std_key = f'{col}_std'
                if mean_key in precomputed_stats and std_key in precomputed_stats:
                    new_cols[f'{col}_grp_mean'] = result['rn'].map(precomputed_stats[mean_key])
                    new_cols[f'{col}_grp_std'] = result['rn'].map(precomputed_stats[std_key])
            if new_cols:
                result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
        else:
            group_stats = df.groupby('rn')[numeric_cols].agg(['mean', 'std', 'min', 'max', 'median']).reset_index()
            precomputed_stats = {}
            new_cols = {}
            for col in numeric_cols:
                for stat in ['mean', 'std', 'min', 'max', 'median']:
                    key = f'{col}_{stat}'
                    if (col, stat) in group_stats.columns:
                        precomputed_stats[key] = dict(zip(group_stats['rn'], group_stats[(col, stat)]))
            for col in numeric_cols:
                for stat in ['mean', 'std', 'min', 'max', 'median']:
                    key = f'{col}_{stat}'
                    if key in precomputed_stats:
                        new_cols[f'{col}_grp_{stat}'] = result['rn'].map(precomputed_stats[key])
            if new_cols:
                result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
        # Взаимодействия для топ-5 признаков
        top_cols = numeric_cols[:5]
        new_features = {}
        for i, col1 in enumerate(top_cols):
            for col2 in top_cols[i+1:]:
                if col1 in result.columns and col2 in result.columns:
                    new_features[f'ratio_{col1}_{col2}'] = result[col1] / (result[col2] + 1e-8)
                    new_features[f'sum_{col1}_{col2}'] = result[col1] + result[col2]
                    new_features[f'diff_{col1}_{col2}'] = result[col1] - result[col2]
                    new_features[f'prod_{col1}_{col2}'] = result[col1] * result[col2]
        if new_features:
            result = pd.concat([result, pd.DataFrame(new_features, index=result.index)], axis=1)
        # Логарифмы и бинаризация (исправлено)
        log_bin_features = {}
        for col in numeric_cols:
            log_bin_features[f'{col}_log1p'] = np.log1p(result[col].abs())
            log_bin_features[f'{col}_bin'] = (result[col] > result[col].median()).astype(int)
        if log_bin_features:
            result = pd.concat([result, pd.DataFrame(log_bin_features, index=result.index)], axis=1)

        return result, precomputed_stats
    except Exception as e:
        logging.error(f"Ошибка при создании признаков: {str(e)}")
        raise

# Глобально инициализируем список признаков и scaler
all_feature_names = None
scaler = StandardScaler()

def process_chunk(chunk_df: pd.DataFrame, target_df: pd.DataFrame = None, precomputed_stats: Dict = None) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    global all_feature_names, scaler
    try:
        if target_df is not None:
            chunk_with_target = chunk_df.merge(target_df, on="id", how='left')
        else:
            chunk_with_target = chunk_df.copy()
        numeric_cols = chunk_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'rn']]
        chunk_with_features, precomputed_stats = create_features(chunk_with_target, numeric_cols, precomputed_stats)
        X = chunk_with_features.drop(columns=["id", "rn"] + (["target"] if "target" in chunk_with_features.columns else []))
        y = chunk_with_features["target"] if "target" in chunk_with_features.columns else None

        # Инициализация признаков и scaler только на первом чанке
        if all_feature_names is None:
            all_feature_names = X.columns.tolist()
            scaler.fit(X)
        # Приведение к общему набору признаков
        missing_cols = [col for col in all_feature_names if col not in X.columns]
        if missing_cols:
            X = pd.concat([X, pd.DataFrame(0, index=X.index, columns=missing_cols)], axis=1)
        X = X[all_feature_names]
        X = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        X = X.fillna(-999)
        return X, y, precomputed_stats
    except Exception as e:
        logging.error(f"Ошибка при обработке чанка: {str(e)}")
        raise

try:
    start_time = time.time()
    logging.info("Загрузка целевых значений...")
    train_target = pd.read_csv("train_target.csv")

    all_files = list(range(12))
    train_files = all_files[:9]
    val_files = all_files[9:]

    logging.info("Начало обучения...")
    best_score = 0
    precomputed_stats = None

    for epoch in range(1):      # было 3
        logging.info(f"\nЭпоха {epoch + 1}/3")
        for i in train_files:
            logging.info(f"Обработка файла {i}/12...")
            df = pd.read_parquet(f"train_data/train_data_{i}.pq")
            X_chunks = []
            y_chunks = []
            for start_idx in range(0, len(df), CHUNK_SIZE):
                chunk = df.iloc[start_idx:start_idx + CHUNK_SIZE]
                X_chunk, y_chunk, precomputed_stats = process_chunk(chunk, train_target, precomputed_stats)
                X_chunks.append(X_chunk)
                y_chunks.append(y_chunk)
                del X_chunk, y_chunk
                gc.collect()
            # Объединяем все чанки файла
            X_file = pd.concat(X_chunks, axis=0)
            y_file = pd.concat(y_chunks, axis=0)
            if epoch == 0 and i == train_files[0]:
                model.fit(X_file, y_file, eval_set=[(X_file, y_file)], eval_metric='auc')
            else:
                model.fit(X_file, y_file, init_model=model, eval_set=[(X_file, y_file)], eval_metric='auc')
            del X_chunks, y_chunks, X_file, y_file, df
            gc.collect()

        val_predictions = []
        val_true = []
        for i in val_files:
            df = pd.read_parquet(f"train_data/train_data_{i}.pq")
            for start_idx in range(0, len(df), CHUNK_SIZE):
                chunk = df.iloc[start_idx:start_idx + CHUNK_SIZE]
                X_chunk, y_chunk, _ = process_chunk(chunk, train_target, precomputed_stats)
                val_predictions.extend(model.predict_proba(X_chunk)[:, 1])
                val_true.extend(y_chunk)
            del df
            gc.collect()

        score = roc_auc_score(val_true, val_predictions)
        logging.info(f"Эпоха {epoch + 1} - ROC-AUC: {score:.4f}")
        print(f"ROC-AUC на валидации: {score:.4f}")

        if score > 0.73:
            print("Поздравляем! ROC-AUC на валидации выше 0.73!")
        else:
            print("ROC-AUC на валидации ниже 0.73. Улучшайте модель и признаки.")

        if score > best_score:
            best_score = score
            joblib.dump(model, "best_model_optimized.pkl")

    # Для финального сабмита обучите модель на всех файлах (train+val)
    # и сделайте предсказания на тесте

    logging.info("\nОбработка тестовых данных...")
    test_predictions = []
    for i in range(2):
        logging.info(f"Тестовый файл {i}/2...")
        df = pd.read_parquet(f"test_data/test_data_{i}.pq")
        chunk_predictions = []
        chunk_ids = []
        for start_idx in range(0, len(df), CHUNK_SIZE):
            chunk = df.iloc[start_idx:start_idx + CHUNK_SIZE]
            X_test_chunk, _, _ = process_chunk(chunk, None, precomputed_stats)
            chunk_pred = model.predict_proba(X_test_chunk)[:, 1]
            chunk_predictions.extend(chunk_pred)
            chunk_ids.extend(chunk["id"].values)
            del X_test_chunk
            gc.collect()
        test_df_pred = pd.DataFrame({"id": chunk_ids, "target": chunk_predictions})
        test_predictions.append(test_df_pred)
        del df, chunk_ids, chunk_predictions
        gc.collect()

    final_predictions = pd.concat(test_predictions, ignore_index=True)
    final_predictions.to_csv("submission_optimized.csv", index=False)

    total_time = time.time() - start_time
    logging.info(f"\nГотово! Время выполнения: {total_time/60:.1f} минут")
    logging.info(f"Лучший ROC-AUC на валидации: {best_score:.4f}")
    logging.info("Результаты сохранены в submission_optimized.csv")
    print(f"\nЛучший ROC-AUC на валидации: {best_score:.4f}")

except Exception as e:
    logging.error(f"Критическая ошибка: {str(e)}")
    raise