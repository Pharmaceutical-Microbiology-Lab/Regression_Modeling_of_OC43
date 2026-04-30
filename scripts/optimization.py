import optuna
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score
from optuna.samplers import TPESampler

def tune_feature_weights(X, y, feature_names, seed, initial_weights=None, n_trials=100, 
                          n_splits=5):
    """
    Optimizes feature weights for CatBoostRegressor using Optuna.

    Parameters
    ----------
    X : pd.DataFrame or np.array
        Training features.
    y : pd.Series or np.array
        Target variable.
    feature_names : list
        List of feature names (genes) corresponding to X columns.
    initial_weights : dict, optional
        Dictionary of initial weights {gene_name: weight} to enqueue as the first trial.
    n_trials : int
        Number of optimization trials.
    seed : int
        Seed for reproducibility.
    n_splits : int
        Number of folds for Cross-Validation.

    Returns
    -------
    best_weights_map : dict
        Dictionary of optimized weights {gene_name: weight}.
    study : optuna.study.Study
        The optimization study object.
    """
    
    # Optuna Objective Function (Closure to capture X, y)
    def objective(trial):
        # 1. Suggest weights
        weights = []
        for gene in feature_names:
            w = trial.suggest_float(f"weight_{gene}", 0.5, 15.0)
            weights.append(w)

        # 2. Define Model
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_state=seed,
            verbose=0,
            feature_weights=weights,
            task_type='CPU',     
            thread_count=-1,
            allow_writing_files=False # Prevent creating catboost_info folder
        )

        # 3. Cross Validation
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return scores.mean()
    
    sampler = TPESampler(seed=seed) 
    # Create Study
    study = optuna.create_study(direction='maximize', sampler=sampler)

    # Enqueue Initial Guess if provided
    if initial_weights:
        # Convert {gene: weight} to {weight_gene: weight} format for Optuna
        trial_params = {f"weight_{k}": v for k, v in initial_weights.items() if k in feature_names}
        if trial_params:
            study.enqueue_trial(trial_params)
            print(f"ℹ️ Enqueued initial weights for {len(trial_params)} features.")

    # Run Optimization
    print(f"🚀 Starting Feature Weight Optimization ({n_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.INFO) # Suppress generic logs
    study.optimize(objective, n_trials=n_trials)
    
    print(f"✅ Optimization Complete. Best R2: {study.best_value:.4f}")

    # Process Best Params into a clean dictionary
    best_weights_map = {}
    for param_key, val in study.best_params.items():
        gene = param_key.replace("weight_", "")
        best_weights_map[gene] = val
        
    return best_weights_map, study


def tune_hyperparameters(X, y, seed, feature_weights=None, n_trials=100, 
                          n_splits=5):
    """
    Optimizes CatBoost hyperparameters (depth, l2_leaf_reg, learning_rate).

    Parameters
    ----------
    X : pd.DataFrame or np.array
        Training features.
    y : pd.Series or np.array
        Target variable.
    feature_weights : list or dict, optional
        Fixed feature weights to apply during tuning. 
        If dict, it is converted to list based on X columns order (if X is DataFrame).
    n_trials : int
        Number of trials.
        
    Returns
    -------
    best_params : dict
        Dictionary of optimized hyperparameters.
    study : optuna.study.Study
        The study object.
    """
    
    # Handle feature weights format
    weights_list = None
    if feature_weights is not None:
        if isinstance(feature_weights, dict) and hasattr(X, 'columns'):
            weights_list = [feature_weights.get(col, 1.0) for col in X.columns]
        elif isinstance(feature_weights, (list, np.ndarray)):
            weights_list = feature_weights
    
    def objective(trial):
        # 1. Suggest Hyperparameters
        depth = trial.suggest_int('depth', 4, 9)
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 0.1, 30.0, log=True)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)

        # 2. Define Model
        model = CatBoostRegressor(
            iterations=1000,
            random_state=seed,
            verbose=0,
            feature_weights=weights_list, # Fixed weights
            task_type='CPU',
            thread_count=-1,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            learning_rate=learning_rate,
            allow_writing_files=False
        )

        # 3. Cross Validation
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return scores.mean()

    print(f"🚀 Starting Hyperparameter Tuning ({n_trials} trials)...")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    sampler = TPESampler(seed=seed) 
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"✅ Tuning Complete. Best R2: {study.best_value:.4f}")
    
    return study.best_params, study