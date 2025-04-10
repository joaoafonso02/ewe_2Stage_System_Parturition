import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from joblib import parallel_backend, Parallel, delayed
import tensorflow as tf
import pickle
from datetime import datetime
import json
import gc
import seaborn as sns
import io
import os
import psutil
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm

# Fix TensorFlow GPU initialization issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # Use only the first GPU

# Configure GPU memory growth - more carefully
print("TensorFlow version:", tf.__version__)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU devices available: {len(gpus)}")
        for gpu in gpus:
            # Limit memory growth
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for GPU: {gpu}")
            except Exception as e:
                print(f"Error setting memory growth: {e}")
        print("GPU memory growth enabled")
    else:
        print("No GPU found, falling back to CPU")
except Exception as e:
    print(f"Error configuring GPUs: {e}")

# Chunk size for data processing - adjust based on sample rate
CHUNK_SIZE = 500000  # Adjust based on your system's memory capacity

SAMPLE_RATES = [
    ('0.5Hz', '2s'),     # 0.5 Hz = 1 sample every 2 seconds
    ('1Hz', '1s'),       # 1 Hz = 1 sample per second
    ('2Hz', '500ms'),    # 2 Hz = 2 samples per second
    #('3Hz', '333ms'),    # 3 Hz = 3 samples per second
]

# Define window sizes to test
WINDOW_SIZES_MINUTES = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
WINDOW_SIZES = [(min, min * 60) for min in WINDOW_SIZES_MINUTES]  # (minutes, seconds)

# Create output directories
os.makedirs('experiment_results', exist_ok=True)
os.makedirs('experiment_results/plots', exist_ok=True)
os.makedirs('experiment_results/models', exist_ok=True)
os.makedirs('experiment_results/models/binary', exist_ok=True)
os.makedirs('experiment_results/models/multiclass', exist_ok=True)

def print_memory_usage():
    """Print current memory usage of the process"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")

def get_adaptive_chunk_size(sample_rate_hz):
    """Get appropriate chunk size based on sample rate"""
    base_chunk_size = 500000
    rate_factor = float(sample_rate_hz.replace('Hz', ''))
    adaptive_size = int(base_chunk_size / rate_factor)
    print(f"Using adaptive chunk size of {adaptive_size} for {sample_rate_hz}")
    return adaptive_size

def dataframe_shift(df, columns, window_seconds, sample_rate_hz):
    """Create windowed features with memory optimization"""
    steps = int(window_seconds * float(sample_rate_hz.replace('Hz', '')))
    print(f"Window would have {steps} steps for {window_seconds}s window at {sample_rate_hz}")
    
    # Use fewer points for higher sampling rates
    rate_factor = float(sample_rate_hz.replace('Hz', ''))
    points_per_feature = max(8, min(16, int(16 / rate_factor)))
    
    if steps <= points_per_feature * len(columns):
        print(f"Using all {steps} points")
        # Add columns in batches to reduce memory pressure
        for i in range(1, steps, max(1, steps // 10)):
            batch_columns = []
            for col in columns:
                batch_columns.append(pl.col(col).shift(i).alias(f'prev_{i}_{col}'))
            df = df.with_columns(batch_columns)
            # Force garbage collection after each batch
            gc.collect()
    else:
        stride = max(1, steps // points_per_feature)
        print(f"Reducing {steps} points to ~{points_per_feature} points per feature with stride {stride}")
        
        for col in columns:
            sample_points = list(range(1, steps, stride))[:points_per_feature]
            batch_size = max(1, len(sample_points) // 4)  # Process in 4 batches
            
            for i in range(0, len(sample_points), batch_size):
                batch_points = sample_points[i:i+batch_size]
                batch_columns = []
                for point in batch_points:
                    batch_columns.append(pl.col(col).shift(point).alias(f'prev_{point}_{col}'))
                df = df.with_columns(batch_columns)
                # Force garbage collection after each batch
                gc.collect()
    
    return df.drop_nulls()

def optimize_threshold_for_class_balance(y_true, y_proba):
    # Target the correct proportion of partum samples
    target_proportion = sum(y_true) / len(y_true)
    
    # Find threshold that gives similar proportion in predictions
    best_threshold = 0.5
    best_diff = float('inf')
    
    for threshold in np.linspace(0.1, 0.9, 41):
        y_pred = (y_proba >= threshold).astype(int)
        pred_proportion = sum(y_pred) / len(y_pred)
        diff = abs(pred_proportion - target_proportion)
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
            
    return best_threshold

def generate_models(n_input, n_output, light=False):
    if light:
        models = [
            ('DecisionTreeClassifier', DecisionTreeClassifier(
            max_depth=244, max_features='sqrt', max_leaf_nodes=850, 
            random_state=42, class_weight='balanced')),
            # ('RandomForestClassifier', RandomForestClassifier(n_estimators=32,
            # max_depth=128, max_features='sqrt', max_leaf_nodes=512, 
            # random_state=42, class_weight='balanced')),
        ]
    else:
        models = [
            ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=42, class_weight='balanced')),
            ('RandomForestClassifier', RandomForestClassifier(random_state=42,  class_weight='balanced')),
            ('ExtraTreesClassifier', ExtraTreesClassifier(random_state=42, class_weight='balanced')),
            ('Bagging', BaggingClassifier(random_state=42)),
        ]

    return models

def train_and_evaluate_model(name, clf, X_train, y_train, X_test, y_test, is_binary=True):
    """Train and evaluate a single model with detailed metrics and visualization"""
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    
    # Make predictions
    if hasattr(clf, "predict_proba") and is_binary:
        y_proba = clf.predict_proba(X_test)[:, 1]
        best_threshold = optimize_threshold_for_class_balance(y_test, y_proba)
        print(f"Best threshold for {name}: {best_threshold:.2f}")
        y_pred = (y_proba >= best_threshold).astype(int)
    else:
        y_pred = clf.predict(X_test)
        best_threshold = None
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    
    # Print metrics
    print(f'{name:<22} {acc:>8.2f} {precision:>9.2f} {recall:>6.2f} {f1:>8.2f} {mcc:>5.2f}')
    
    # Get model size
    with io.BytesIO() as buffer:
        pickle.dump(clf, buffer)
        model_size_kb = buffer.getbuffer().nbytes / 1024
    
    return {
        'model': clf,
        'name': name,
        'predictions': y_pred,
        'threshold': best_threshold,
        'metrics': (acc, precision, recall, f1, mcc),
        'size_kb': model_size_kb
    }

def process_data_in_chunks(df_windowed, sample_rate, window_size, unique_labels, experiment_results, aggregate_data):
    """Process windowed data in manageable chunks to avoid memory issues"""
    print(f'--- Processing data - Sample Rate: {sample_rate} - Window: {window_size[0]}min ---')
    
    # Create binary labels (partum vs non-partum)
    df_windowed = df_windowed.with_columns(
        pl.when(pl.col('Class') < 13).then(1).otherwise(0).alias('Binary_Class')
    )
    
    # Analyze class distribution
    binary_distribution = df_windowed['Binary_Class'].value_counts()
    print(f"Original binary class distribution: {binary_distribution}")
    
    # First split into train/test to avoid data leakage
    train_indices, test_indices = train_test_split(
        np.arange(len(df_windowed)), test_size=0.2, random_state=42, 
        stratify=df_windowed['Binary_Class'].to_numpy()
    )
    
    # Extract test set
    # Extract test set
    X_test_full = df_windowed.drop(['Class', 'Binary_Class', 'Time']).to_numpy()[test_indices]
    y_test_full = df_windowed['Class'].to_numpy()[test_indices]
    y_binary_test_full = df_windowed['Binary_Class'].to_numpy()[test_indices]

    # Get partum and non-partum test indices
    test_partum_indices = np.where(y_binary_test_full == 1)[0]
    test_non_partum_indices = np.where(y_binary_test_full == 0)[0]

    # Balance test set by undersampling non-partum to match partum
    if len(test_non_partum_indices) > len(test_partum_indices):
        test_non_partum_indices = np.random.choice(test_non_partum_indices, len(test_partum_indices), replace=False)

    # Create balanced test set
    test_balanced_indices = np.concatenate([test_partum_indices, test_non_partum_indices])
    np.random.shuffle(test_balanced_indices)

    # Create balanced test datasets
    X_test = X_test_full[test_balanced_indices]
    y_test = y_test_full[test_balanced_indices]
    y_binary_test = y_binary_test_full[test_balanced_indices]

    print(f"Test dataset size: {len(y_binary_test)} samples")
    print(f"  - Partum samples: {len(test_partum_indices)}")
    print(f"  - Non-partum samples: {len(test_non_partum_indices)}")

    # Get only partum samples for the multiclass model (from the full test set)
    partum_mask_test = y_binary_test_full == 1
    X_test_partum = X_test_full[partum_mask_test]
    y_test_partum = y_test_full[partum_mask_test]
    
    # Get training indices for partum/non-partum
    train_partum_indices = train_indices[df_windowed['Binary_Class'].to_numpy()[train_indices] == 1]
    train_non_partum_indices = train_indices[df_windowed['Binary_Class'].to_numpy()[train_indices] == 0]

    # Combine indices (unbalanced)
    train_balanced_indices = np.concatenate([train_partum_indices, train_non_partum_indices])
    np.random.shuffle(train_balanced_indices)

    print(f"Training dataset size: {len(train_balanced_indices)} samples")
    print(f"  - Partum samples: {len(train_partum_indices)}")
    print(f"  - Non-partum samples: {len(train_non_partum_indices)}")
    
    # Extract training data in chunks
    X_columns = [col for col in df_windowed.columns if col not in ['Class', 'Binary_Class', 'Time']]
    
    # Process binary classification
    print("\n=== STAGE 1: Binary Classification (Partum vs Non-Partum) ===")
    print(f'{"":<22} Accuracy Precision Recall F1-score   MCC')
    
    # Extract binary data
    chunk_size = min(get_adaptive_chunk_size(sample_rate), len(train_balanced_indices))
    X_train_binary = []
    y_binary_train = []
    
    # Load training data in chunks
    for i in range(0, len(train_balanced_indices), chunk_size):
        chunk_indices = train_balanced_indices[i:i+chunk_size]
        X_chunk = df_windowed.select(X_columns).to_numpy()[chunk_indices]
        y_chunk = df_windowed['Binary_Class'].to_numpy()[chunk_indices]
        
        X_train_binary.append(X_chunk)
        y_binary_train.append(y_chunk)
    
    # Combine chunks for training
    X_train_binary = np.vstack(X_train_binary)
    y_binary_train = np.hstack(y_binary_train)

    # Train binary models
    n_input_binary = X_train_binary.shape[1]
    n_output_binary = 2  # Binary classification
    binary_models = generate_models(n_input_binary, n_output_binary, light=True)
    
    # Train each binary model
    binary_model_results = []
    for name, clf in binary_models:
        with parallel_backend('loky', n_jobs=4):
            result = train_and_evaluate_model(
                name, clf, X_train_binary, y_binary_train, 
                X_test, y_binary_test, is_binary=True
            )
            binary_model_results.append(result)
        
        # Create confusion matrix
        cm = confusion_matrix(y_binary_test, result['predictions'])
        cm_percent = cm / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues', cbar=False)
        plt.title(f"Binary Confusion Matrix - {name} - MCC: {result['metrics'][4]:.2f}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks([0.5, 1.5], ['Non-Partum', 'Partum'])
        plt.yticks([0.5, 1.5], ['Non-Partum', 'Partum'])
        #plt.savefig(f"experiment_results/plots/{name}_confusion_matrix_{sample_rate}_{window_size[0]}min.png")
        plt.close()
        
        # Save binary model
        model_path = f"experiment_results/models/binary/{name}_{sample_rate}_{window_size[0]}min.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Clean up to save memory
        gc.collect()
    
    # Select best binary model
    best_binary_result = max(binary_model_results, key=lambda x: x['metrics'][4])  # Sort by MCC
    
    # Add binary predictions to aggregate data
    aggregate_data['binary_true'].extend(y_binary_test.tolist())
    aggregate_data['binary_pred'].extend(best_binary_result['predictions'].tolist())
    
    # Store the best binary model if it's better than what we have or if we don't have one yet
    if ('best_binary_model' not in aggregate_data or 
        'best_binary_mcc' not in aggregate_data or 
        best_binary_result['metrics'][4] > aggregate_data['best_binary_mcc']):
        
        aggregate_data['best_binary_model'] = best_binary_result['model']
        aggregate_data['best_binary_mcc'] = best_binary_result['metrics'][4]
        aggregate_data['best_binary_name'] = best_binary_result['name']
        print(f"New best binary model: {best_binary_result['name']} with MCC: {best_binary_result['metrics'][4]:.3f}")
    
    # Load partum training data for multiclass in chunks
    X_train_partum = []
    y_train_partum = []
    
    print("\nLoading partum training data for multiclass...", flush=True)
    
    # Extract partum training data in chunks
    chunk_size = min(get_adaptive_chunk_size(sample_rate), len(train_partum_indices))
    for i in range(0, len(train_partum_indices), chunk_size):
        chunk_indices = train_partum_indices[i:i+chunk_size]
        X_chunk = df_windowed.select(X_columns).to_numpy()[chunk_indices]
        y_chunk = df_windowed['Class'].to_numpy()[chunk_indices]
        
        X_train_partum.append(X_chunk)
        y_train_partum.append(y_chunk)
    
    # Combine chunks for multiclass training
    X_train_partum = np.vstack(X_train_partum)
    y_train_partum = np.hstack(y_train_partum)
    
    # Initialize multiclass variables
    multiclass_result = None
    
    # Free up memory before multiclass training
    del X_train_binary, y_binary_train
    print_memory_usage()
    gc.collect()
    print_memory_usage()
    
    # Train multiclass models if we have partum samples
    if len(X_train_partum) > 0 and len(X_test_partum) > 0:
        print("\n=== STAGE 2: Multiclass Classification (Hours until Partum) ===")
        print(f'{"":<22} Accuracy Precision Recall F1-score   MCC')
        
        n_input_multi = X_train_partum.shape[1]
        n_output_multi = len(np.unique(y_train_partum))
        
        multiclass_models = generate_models(n_input_multi, n_output_multi, light=False)
        multiclass_model_results = []
        
        # Train each multiclass model
        for mc_name, mc_clf in multiclass_models:
            try:
                with parallel_backend('loky', n_jobs=4):
                    result = train_and_evaluate_model(
                        mc_name, mc_clf, X_train_partum, y_train_partum, 
                        X_test_partum, y_test_partum, is_binary=False
                    )
                    multiclass_model_results.append(result)
                
                # Save multiclass model
                model_path = f"experiment_results/models/multiclass/{mc_name}_{sample_rate}_{window_size[0]}min.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(result['model'], f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Clean up immediately
                gc.collect()
                
            except (MemoryError, RuntimeError) as e:
                print(f"Memory error training {mc_name}: {e}")
                continue
        
        # Select best multiclass model if any were trained successfully
        if multiclass_model_results:
            multiclass_result = max(multiclass_model_results, key=lambda x: x['metrics'][4])  # Sort by MCC
            
            # Add multiclass predictions to aggregate data
            aggregate_data['multiclass_true'].extend(y_test_partum.tolist())
            aggregate_data['multiclass_pred'].extend(multiclass_result['predictions'].tolist())
            
            # Store the best multiclass model if it's better than what we have or if we don't have one yet
            if ('best_multiclass_model' not in aggregate_data or 
                'best_multiclass_mcc' not in aggregate_data or 
                multiclass_result['metrics'][4] > aggregate_data['best_multiclass_mcc']):
                
                aggregate_data['best_multiclass_model'] = multiclass_result['model']
                aggregate_data['best_multiclass_mcc'] = multiclass_result['metrics'][4]
                aggregate_data['best_multiclass_name'] = multiclass_result['name']
                print(f"New best multiclass model: {multiclass_result['name']} with MCC: {multiclass_result['metrics'][4]:.3f}")

        # plot confusion matrix for the best multiclass model
        if multiclass_result:
            cm = confusion_matrix(y_test_partum, multiclass_result['predictions'])
            cm_percent = cm / cm.sum(axis=1)[:, np.newaxis] * 100
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues', cbar=False)
            plt.title(f"Multiclass Confusion Matrix - {multiclass_result['name']} - MCC: {multiclass_result['metrics'][4]:.2f}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(range(len(unique_labels)), unique_labels, rotation=45)
            plt.yticks(range(len(unique_labels)), unique_labels)
            #plt.savefig(f"experiment_results/plots/{multiclass_result['name']}_confusion_matrix_{sample_rate}_{window_size[0]}min.png")
            plt.close()
    
    # Evaluate combined system if we have both models
    combined_metrics = None
    if best_binary_result and multiclass_result:
        print("\n=== COMBINED SYSTEM EVALUATION ===")
        
        # Initialize predictions array with default non-partum class (13)
        y_combined_pred = np.ones_like(y_test) * 13
        
        # Get indices where best binary model predicts partum
        partum_pred_indices = np.where(best_binary_result['predictions'] == 1)[0]
        
        # For those indices, use the best multiclass model
        if len(partum_pred_indices) > 0:
            mc_predictions = multiclass_result['model'].predict(X_test[partum_pred_indices])
            y_combined_pred[partum_pred_indices] = mc_predictions
        
        # Add combined predictions to aggregate data
        aggregate_data['all_true'].extend(y_test.tolist())
        aggregate_data['combined_pred'].extend(y_combined_pred.tolist())
        
        # Calculate combined metrics
        combined_acc = accuracy_score(y_test, y_combined_pred)
        combined_precision = precision_score(y_test, y_combined_pred, average='weighted')
        combined_recall = recall_score(y_test, y_combined_pred, average='weighted')
        combined_f1 = f1_score(y_test, y_combined_pred, average='weighted')
        combined_mcc = matthews_corrcoef(y_test, y_combined_pred)
        
        combined_metrics = (combined_acc, combined_precision, combined_recall, combined_f1, combined_mcc)
        
        print(f'Combined System     {combined_acc:>8.2f} {combined_precision:>9.2f} {combined_recall:>6.2f} {combined_f1:>8.2f} {combined_mcc:>5.2f}')

        # Store the best combined system if it's better than what we have or if we don't have one yet
        if ('best_combined_mcc' not in aggregate_data or combined_mcc > aggregate_data['best_combined_mcc']):
            aggregate_data['best_combined_mcc'] = combined_mcc
            aggregate_data['best_combined_binary_model'] = best_binary_result['model']
            aggregate_data['best_combined_multiclass_model'] = multiclass_result['model']
            aggregate_data['best_combined_binary_name'] = best_binary_result['name']
            aggregate_data['best_combined_multiclass_name'] = multiclass_result['name']
            print(f"New best combined system: {best_binary_result['name']} + {multiclass_result['name']} with MCC: {combined_mcc:.3f}")

        # plot combined confusion matrix
        cm = confusion_matrix(y_test, y_combined_pred)
        cm_percent = cm / cm.sum(axis=1)[:, np.newaxis] * 100
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues', cbar=False)
        plt.title(f"Combined Confusion Matrix - MCC: {combined_mcc:.2f}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(len(unique_labels)), unique_labels, rotation=45)
        plt.yticks(range(len(unique_labels)), unique_labels)
        #plt.savefig(f"experiment_results/plots/combined_confusion_matrix_{sample_rate}_{window_size[0]}min.png")
    
    # Write results to CSV
    csv_path = 'experiment_results/model_performance.csv'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('sample_rate,window_minutes,binary_model,binary_accuracy,binary_precision,binary_recall,' +
                   'binary_f1,binary_mcc,binary_threshold,binary_size_kb,multiclass_model,multiclass_accuracy,' +
                   'multiclass_precision,multiclass_recall,multiclass_f1,multiclass_mcc,combined_accuracy,' +
                   'combined_precision,combined_recall,combined_f1,combined_mcc\n')
    
    # Format values for CSV
    def format_value(value):
        if value is None:
            return ""
        return f"{float(value):.6f}"
    
    # Get metrics from results
    binary_metrics = best_binary_result['metrics']
    mc_metrics = multiclass_result['metrics'] if multiclass_result else (None, None, None, None, None)
    
    # Add row to CSV
    with open(csv_path, 'a') as f:
        row = [
            sample_rate,
            window_size[0],
            best_binary_result['name'],
            format_value(binary_metrics[0]),  # accuracy
            format_value(binary_metrics[1]),  # precision
            format_value(binary_metrics[2]),  # recall
            format_value(binary_metrics[3]),  # f1
            format_value(binary_metrics[4]),  # mcc
            format_value(best_binary_result['threshold']),
            format_value(best_binary_result['size_kb']),
            multiclass_result['name'] if multiclass_result else "",
            format_value(mc_metrics[0]),  # accuracy
            format_value(mc_metrics[1]),  # precision
            format_value(mc_metrics[2]),  # recall
            format_value(mc_metrics[3]),  # f1
            format_value(mc_metrics[4]),  # mcc
            format_value(combined_metrics[0] if combined_metrics else None),  # combined accuracy
            format_value(combined_metrics[1] if combined_metrics else None),  # combined precision
            format_value(combined_metrics[2] if combined_metrics else None),  # combined recall
            format_value(combined_metrics[3] if combined_metrics else None),  # combined f1
            format_value(combined_metrics[4] if combined_metrics else None)   # combined mcc
        ]
        f.write(",".join(map(str, row)) + "\n")
    
    # Save experimental result for JSON output
    result_summary = {
        'sample_rate': sample_rate,
        'window_minutes': window_size[0],
        'binary_model': best_binary_result['name'],
        'binary_mcc': float(binary_metrics[4]),
    }
    
    if multiclass_result:
        result_summary.update({
            'multiclass_model': multiclass_result['name'],
            'multiclass_mcc': float(mc_metrics[4]),
        })
    
    if combined_metrics:
        result_summary['combined_mcc'] = float(combined_metrics[4])
        result_summary['binary_model_for_combined'] = best_binary_result['name']
        result_summary['multiclass_model_for_combined'] = multiclass_result['name']
    
    experiment_results.append(result_summary)
    
    print(f"Results written to CSV: {csv_path}")
    gc.collect()
    return experiment_results

def process_files_for_config(rate_hz, rate_interval, window_min, window_sec, unique_labels, experiment_results):
    """Process all files for one configuration (sample rate and window size)"""
    print(f"\n--- Testing Sample Rate: {rate_hz}, Window Size: {window_min} minutes ---")
    
    all_files = os.listdir('data/train2')    
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
    
    # Create combined results dataframe after windowing
    all_results = None
    
    # Initialize aggregate data storage
    aggregate_data = {
        'binary_true': [],
        'binary_pred': [],
        'multiclass_true': [],
        'multiclass_pred': [],
        'all_true': [],
        'combined_pred': []
    }
    
    # Process each file separately and completely
    for dataset in tqdm(csv_files, desc="Processing files"):
        print(f"Processing {dataset}...")
        
        # Load and resample file
        df = pl.read_csv(f'data/train2/{dataset}', separator=';')
        
        df = df.with_columns(
            pl.col('Time').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S.%3f')
        )
        
        # Resample at current rate
        df_resampled = df.set_sorted('Time').group_by_dynamic('Time', every=rate_interval).agg(
            pl.col('Acc_X (mg)').median(),
            pl.col('Acc_Y (mg)').median(),
            pl.col('Acc_Z (mg)').median(),
            pl.col('Temperature (C)').median(),
            pl.col('Class').mode().first()
        )
        
        # Scale
        df_resampled = df_resampled.with_columns(
            pl.col('Acc_X (mg)').map_batches(lambda x: pl.Series(minmax_scale(x))),
            pl.col('Acc_Y (mg)').map_batches(lambda x: pl.Series(minmax_scale(x))),
            pl.col('Acc_Z (mg)').map_batches(lambda x: pl.Series(minmax_scale(x)))
        )
        
        # Create windowed features
        df_windowed = dataframe_shift(
            df_resampled, 
            columns=['Acc_X (mg)', 'Acc_Y (mg)', 'Acc_Z (mg)', 'Temperature (C)'], 
            window_seconds=window_sec,
            sample_rate_hz=rate_hz
        )
        
        # Add to combined results or process immediately
        if all_results is None:
            all_results = df_windowed
        else:
            all_results = pl.concat([all_results, df_windowed])
        
        # Process in chunks if the combined dataset is getting too large
        chunk_size = get_adaptive_chunk_size(rate_hz)
        if all_results.shape[0] > chunk_size * 2:
            print(f"Processing accumulated data batch (rows: {all_results.shape[0]})...")
            experiment_results = process_data_in_chunks(
                all_results, 
                sample_rate=rate_hz,
                window_size=(window_min, window_sec),
                unique_labels=unique_labels,
                experiment_results=experiment_results,
                aggregate_data=aggregate_data
            )
            # Reset accumulated results
            all_results = None
            gc.collect()
        
        # Free memory
        del df, df_resampled, df_windowed
        gc.collect()
    
    # Process any remaining data
    if all_results is not None and all_results.shape[0] > 0:
        print(f"Processing final data batch (rows: {all_results.shape[0]})...")
        experiment_results = process_data_in_chunks(
            all_results, 
            sample_rate=rate_hz,
            window_size=(window_min, window_sec),
            unique_labels=unique_labels,
            experiment_results=experiment_results,
            aggregate_data=aggregate_data
        )
    
    # Create aggregate confusion matrices and plots
    create_aggregate_confusion_matrices(aggregate_data, rate_hz, window_min, unique_labels)
    
    return experiment_results

def create_aggregate_confusion_matrices(aggregate_data, rate_hz, window_min, unique_labels):
    """Create confusion matrices for aggregate predictions across all files"""
    print("\n=== Creating aggregate confusion matrices ===")
    
    # Binary confusion matrix
    if aggregate_data['binary_true'] and aggregate_data['binary_pred']:
        binary_cm = confusion_matrix(
            aggregate_data['binary_true'], 
            aggregate_data['binary_pred']
        )

        binary_mcc = matthews_corrcoef(
            aggregate_data['binary_true'], 
            aggregate_data['binary_pred']
        )
        
        # # Display both counts and percentages
        plt.figure(figsize=(8, 6))
        # Format annotations to include both count and percentage
        fmt_cm = np.empty_like(binary_cm, dtype=object)
        row_sums = binary_cm.sum(axis=1)
        for i in range(binary_cm.shape[0]):
            for j in range(binary_cm.shape[1]):
                percentage = (binary_cm[i, j] / row_sums[i] * 100) if row_sums[i] > 0 else 0
                fmt_cm[i, j] = f"{binary_cm[i, j]}\n({percentage:.1f}%)"

        # Use raw counts for coloring but show formatted annotations
        sns.heatmap(binary_cm, annot=fmt_cm, fmt="", cmap='Blues', cbar=False)
        # Change how you display confusion matrices
        # plt.figure(figsize=(8, 6))
        # cm_percent = binary_cm / binary_cm.sum(axis=1)[:, np.newaxis] * 100
        # sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues', cbar=False)
        plt.title(f"Aggregate Binary Confusion Matrix - {rate_hz} - {window_min}min -> MCC: {binary_mcc:.2f}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks([0.5, 1.5], ['Non-Partum', 'Partum'])
        plt.yticks([0.5, 1.5], ['Non-Partum', 'Partum'])
        plt.savefig(f"experiment_results/plots/aggregate_binary_cm_{rate_hz}_{window_min}min.png")
        plt.close()
                
        # Calculate overall binary metrics
        binary_acc = accuracy_score(aggregate_data['binary_true'], aggregate_data['binary_pred'])
        binary_mcc = matthews_corrcoef(aggregate_data['binary_true'], aggregate_data['binary_pred'])
        binary_f1 = f1_score(aggregate_data['binary_true'], aggregate_data['binary_pred'], average='weighted')
        
        print(f"Aggregate Binary Results: Acc={binary_acc:.3f}, MCC={binary_mcc:.3f}, F1={binary_f1:.3f}")
        
        # Save aggregate metrics to a JSON file
        aggregate_metrics = {
            'binary': {
                'accuracy': float(binary_acc),
                'mcc': float(binary_mcc),
                'f1': float(binary_f1),
                'confusion_matrix': binary_cm.tolist()
            }
        }
        
        # If we have a 'best_binary_model' in aggregate_data, save it
        if 'best_binary_model' in aggregate_data:
            # Save the best binary model
            model_path = f"experiment_results/models/binary/best_aggregate_{rate_hz}_{window_min}min.pkl"
            print(f"Saving best binary model to {model_path}")
            with open(model_path, 'wb') as f:
                pickle.dump(aggregate_data['best_binary_model'], f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Multiclass confusion matrix
    if aggregate_data['multiclass_true'] and aggregate_data['multiclass_pred']:
        # Get unique classes in the multiclass data
        unique_mc_classes = sorted(list(set(aggregate_data['multiclass_true'] + aggregate_data['multiclass_pred'])))
        
        multi_cm = confusion_matrix(
            aggregate_data['multiclass_true'], 
            aggregate_data['multiclass_pred'],
            labels=unique_mc_classes
        )
        
        # Handle potential class imbalance
        row_sums = multi_cm.sum(axis=1)
        multi_cm_percent = np.zeros_like(multi_cm, dtype=float)
        for i, row_sum in enumerate(row_sums):
            if row_sum > 0:
                multi_cm_percent[i] = multi_cm[i] / row_sum * 100

        plt.figure(figsize=(10, 8))
        sns.heatmap(multi_cm_percent, annot=True, fmt=".1f", cmap='Blues', cbar=False)
        plt.title(f"Aggregate Multiclass Confusion Matrix - {rate_hz} - {window_min}min")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(np.arange(len(unique_mc_classes)) + 0.5, unique_mc_classes, rotation=45)
        plt.yticks(np.arange(len(unique_mc_classes)) + 0.5, unique_mc_classes)
        plt.savefig(f"experiment_results/plots/aggregate_multiclass_cm_{rate_hz}_{window_min}min.png")
        plt.close()
        
        # Calculate overall multiclass metrics
        multi_acc = accuracy_score(aggregate_data['multiclass_true'], aggregate_data['multiclass_pred'])
        multi_mcc = matthews_corrcoef(aggregate_data['multiclass_true'], aggregate_data['multiclass_pred'])
        multi_f1 = f1_score(aggregate_data['multiclass_true'], aggregate_data['multiclass_pred'], average='weighted')
        
        print(f"Aggregate Multiclass Results: Acc={multi_acc:.3f}, MCC={multi_mcc:.3f}, F1={multi_f1:.3f}")
        
        # Add multiclass metrics to aggregate_metrics
        if 'aggregate_metrics' not in locals():
            aggregate_metrics = {}
        
        aggregate_metrics['multiclass'] = {
            'accuracy': float(multi_acc),
            'mcc': float(multi_mcc),
            'f1': float(multi_f1),
            'confusion_matrix': multi_cm.tolist()
        }
        
        # If we have a 'best_multiclass_model' in aggregate_data, save it
        if 'best_multiclass_model' in aggregate_data:
            # Save the best multiclass model
            model_path = f"experiment_results/models/multiclass/best_aggregate_{rate_hz}_{window_min}min.pkl"
            print(f"Saving best multiclass model to {model_path}")
            with open(model_path, 'wb') as f:
                pickle.dump(aggregate_data['best_multiclass_model'], f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Combined system confusion matrix
    if aggregate_data['all_true'] and aggregate_data['combined_pred']:
        # Get unique classes in the combined data
        unique_combined_classes = sorted(list(set(aggregate_data['all_true'] + aggregate_data['combined_pred'])))
        
        combined_cm = confusion_matrix(
            aggregate_data['all_true'], 
            aggregate_data['combined_pred'],
            labels=unique_combined_classes
        )

        combined_mcc = matthews_corrcoef(
            aggregate_data['all_true'], 
            aggregate_data['combined_pred']
        )
        
        # Handle potential class imbalance
        row_sums = combined_cm.sum(axis=1)
        combined_cm_percent = np.zeros_like(combined_cm, dtype=float)
        for i, row_sum in enumerate(row_sums):
            if row_sum > 0:
                combined_cm_percent[i] = combined_cm[i] / row_sum * 100

        plt.figure(figsize=(10, 8))
        sns.heatmap(combined_cm_percent, annot=True, fmt=".1f", cmap='Blues', cbar=False)
        plt.title(f"Aggregate Combined System Matrix - {rate_hz} - {window_min}min -> MCC: {combined_mcc:.2f}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(np.arange(len(unique_combined_classes)) + 0.5, unique_combined_classes, rotation=45)
        plt.yticks(np.arange(len(unique_combined_classes)) + 0.5, unique_combined_classes)
        plt.savefig(f"experiment_results/plots/aggregate_combined_cm_{rate_hz}_{window_min}min.png")
        plt.close()
        
        # Calculate overall combined metrics
        combined_acc = accuracy_score(aggregate_data['all_true'], aggregate_data['combined_pred'])
        combined_mcc = matthews_corrcoef(aggregate_data['all_true'], aggregate_data['combined_pred'])
        combined_f1 = f1_score(aggregate_data['all_true'], aggregate_data['combined_pred'], average='weighted')
        
        print(f"Aggregate Combined Results: Acc={combined_acc:.3f}, MCC={combined_mcc:.3f}, F1={combined_f1:.3f}")
        
        # Add combined metrics to aggregate_metrics
        if 'aggregate_metrics' not in locals():
            aggregate_metrics = {}
        
        aggregate_metrics['combined'] = {
            'accuracy': float(combined_acc),
            'mcc': float(combined_mcc),
            'f1': float(combined_f1),
            'confusion_matrix': combined_cm.tolist()
        }
    
    # Save aggregate metrics to a JSON file
    if 'aggregate_metrics' in locals():
        metrics_path = f"experiment_results/metrics_{rate_hz}_{window_min}min.json"
        with open(metrics_path, 'w') as f:
            json.dump(aggregate_metrics, f)
        
        # Also save to comprehensive CSV
        csv_path = 'experiment_results/aggregate_metrics.csv'
        is_new_csv = not os.path.exists(csv_path)
        
        # Calculate all required metrics for each model type
        binary_precision = precision_score(aggregate_data['binary_true'], aggregate_data['binary_pred'], average='weighted') if 'binary' in aggregate_metrics else None
        binary_recall = recall_score(aggregate_data['binary_true'], aggregate_data['binary_pred'], average='weighted') if 'binary' in aggregate_metrics else None
        
        multiclass_precision = precision_score(aggregate_data['multiclass_true'], aggregate_data['multiclass_pred'], average='weighted') if 'multiclass' in aggregate_metrics else None
        multiclass_recall = recall_score(aggregate_data['multiclass_true'], aggregate_data['multiclass_pred'], average='weighted') if 'multiclass' in aggregate_metrics else None
        
        combined_precision = precision_score(aggregate_data['all_true'], aggregate_data['combined_pred'], average='weighted') if 'combined' in aggregate_metrics else None
        combined_recall = recall_score(aggregate_data['all_true'], aggregate_data['combined_pred'], average='weighted') if 'combined' in aggregate_metrics else None
        
        # Get model names from aggregate_data
        binary_model = aggregate_data.get('best_binary_name', '') if 'binary' in aggregate_metrics else ''
        multiclass_model = aggregate_data.get('best_multiclass_name', '') if 'multiclass' in aggregate_metrics else ''
        
        # Get model size if available
        binary_size_kb = 0
        if 'best_binary_model' in aggregate_data:
            with io.BytesIO() as buffer:
                pickle.dump(aggregate_data['best_binary_model'], buffer)
                binary_size_kb = buffer.getbuffer().nbytes / 1024
        
        # Format values for CSV
        def format_value(value):
            if value is None:
                return ""
            return f"{float(value):.6f}"
        
        # Write to CSV
        with open(csv_path, 'a') as f:
            # Write header if new file
            if is_new_csv:
                f.write('sample_rate,window_minutes,binary_model,binary_accuracy,binary_precision,binary_recall,' +
                    'binary_f1,binary_mcc,binary_threshold,binary_size_kb,multiclass_model,multiclass_accuracy,' +
                    'multiclass_precision,multiclass_recall,multiclass_f1,multiclass_mcc,combined_accuracy,' +
                    'combined_precision,combined_recall,combined_f1,combined_mcc\n')
            
            # Prepare row data
            row = [
                rate_hz,
                window_min,
                binary_model,
                format_value(aggregate_metrics.get('binary', {}).get('accuracy')),
                format_value(binary_precision),
                format_value(binary_recall),
                format_value(aggregate_metrics.get('binary', {}).get('f1')),
                format_value(aggregate_metrics.get('binary', {}).get('mcc')),
                format_value(aggregate_data.get('best_binary_threshold', None)),
                format_value(binary_size_kb),
                multiclass_model,
                format_value(aggregate_metrics.get('multiclass', {}).get('accuracy')),
                format_value(multiclass_precision),
                format_value(multiclass_recall),
                format_value(aggregate_metrics.get('multiclass', {}).get('f1')),
                format_value(aggregate_metrics.get('multiclass', {}).get('mcc')),
                format_value(aggregate_metrics.get('combined', {}).get('accuracy')),
                format_value(combined_precision),
                format_value(combined_recall),
                format_value(aggregate_metrics.get('combined', {}).get('f1')),
                format_value(aggregate_metrics.get('combined', {}).get('mcc'))
            ]
            
            # Write row to CSV
            f.write(",".join(map(str, row)) + "\n")

def run_experiment():
    experiment_results = []
    
    # Process each combination of sample rate and window size
    for rate_hz, rate_interval in SAMPLE_RATES:
        # Get unique labels (do this once outside the loops)
        sample_df = pl.read_csv(f'data/train2/{os.listdir("data/train2")[0]}', separator=';')
        unique = sample_df.unique(subset=['Class'], maintain_order=True)
        unique_labels = sorted(unique['Class'].to_list())
        print("Unique classes found:", unique_labels)
        
        for window_min, window_sec in WINDOW_SIZES:
            # Process one file at a time for this combination
            experiment_results = process_files_for_config(
                rate_hz, 
                rate_interval, 
                window_min, 
                window_sec, 
                unique_labels, 
                experiment_results
            )
            
            # Save intermediate results after each window size
            print("Saving intermediate results...")
            with open(f'experiment_results/results_{rate_hz}_{window_min}min.json', 'w') as f:
                json.dump(experiment_results, f)
    
    # Save final results
    print("Saving final results...")
    with open('experiment_results/results.json', 'w') as f:
        json.dump(experiment_results, f)
    
    try:
        results_df = pd.read_csv('experiment_results/model_performance.csv')
        
        # Find best configuration for each model type
        best_binary = results_df.loc[results_df['binary_mcc'].idxmax()]
        best_multiclass = results_df.loc[results_df['multiclass_mcc'].idxmax()] if 'multiclass_mcc' in results_df.columns else None
        best_combined = results_df.loc[results_df['combined_mcc'].idxmax()] if 'combined_mcc' in results_df.columns else None
        
        best_configs = {
            'binary': {
                'sample_rate': best_binary['sample_rate'],
                'window_minutes': int(best_binary['window_minutes']),
                'model': best_binary['binary_model'],
                'mcc': float(best_binary['binary_mcc']),
                'accuracy': float(best_binary['binary_accuracy']),
                'model_path': f"experiment_results/models/binary/best_aggregate_{best_binary['sample_rate']}_{int(best_binary['window_minutes'])}min.pkl"
            }
        }
        
        if best_multiclass is not None:
            best_configs['multiclass'] = {
                'sample_rate': best_multiclass['sample_rate'],
                'window_minutes': int(best_multiclass['window_minutes']),
                'model': best_multiclass['multiclass_model'],
                'mcc': float(best_multiclass['multiclass_mcc']),
                'accuracy': float(best_multiclass['multiclass_accuracy']),
                'model_path': f"experiment_results/models/multiclass/best_aggregate_{best_multiclass['sample_rate']}_{int(best_multiclass['window_minutes'])}min.pkl"
            }
        
        if best_combined is not None:
            best_configs['combined'] = {
                'sample_rate': best_combined['sample_rate'],
                'window_minutes': int(best_combined['window_minutes']),
                'binary_model': best_combined['binary_model'],
                'multiclass_model': best_combined['multiclass_model'],
                'mcc': float(best_combined['combined_mcc']),
                'accuracy': float(best_combined['combined_accuracy'])
            }
        
        # Save best configurations
        with open('experiment_results/best_configurations.json', 'w') as f:
            json.dump(best_configs, f, indent=2)
        
        print("Best configurations saved to experiment_results/best_configurations.json")
        
    except Exception as e:
        print(f"Error creating best configurations summary: {e}")
        import traceback
        traceback.print_exc()
    
    print("Done!")

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Experiment started at: {start_time}")
    
    try:
        run_experiment()
    except Exception as e:
        print(f"Error in experiment: {e}")
        import traceback
        traceback.print_exc()
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Experiment completed at: {end_time}")
    print(f"Total duration: {duration}")
