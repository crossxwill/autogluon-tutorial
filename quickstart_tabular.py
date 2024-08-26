# %%
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

import numpy as np
import pandas as pd

from sklearnex import patch_sklearn
patch_sklearn()

label = 'signature'
time_limit = 10*60       # max training time (seconds)
infer_limit = 1/10000    # prediction seconds per row
finalModel = 'WeightedEnsemble_L2'

data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'

train_data = TabularDataset(f'{data_url}train.csv')
test_data = TabularDataset(f'{data_url}test.csv')

pruning_test_data = test_data.sample(frac=0.5, random_state=2024)
final_test_data = test_data.drop(pruning_test_data.index)

train_data.head()

# %% 

# Detect features to prune
np.random.seed(2024)

pruning_predictor = TabularPredictor(label=label, problem_type='multiclass', eval_metric='log_loss', log_to_file=True)
pruning_predictor.fit(train_data, presets='medium_quality', time_limit = time_limit)  

pruning_leaders = pruning_predictor.leaderboard(pruning_test_data)
pruning_leaders

# %%

df_pruning_features = pruning_predictor.feature_importance(pruning_test_data)
low_importance_features = df_pruning_features.query("importance < 0.001").index.to_list()
low_importance_features

# %%

final_train_data = train_data.drop(columns=low_importance_features)

# %%
custom_hyperparameters = get_hyperparameter_config('zeroshot') # ['default', 'zeroshot', 'light', 'very_light', 'toy', 'multimodal']

custom_hyperparameters['LR'] = [
    {'multi_class':'multinomial', 'penalty':None, 'tol':1e-6, 'max_iter':10000,
        'ag_args': {'name_suffix': 'Base'}},
    {'multi_class':'multinomial', 'penalty':'l2', 'tol':1e-6, 'max_iter':10000, 'C':0.1,
        'ag_args': {'name_suffix': 'Ridge'}},
    {'multi_class':'multinomial', 'penalty':'l1', 'tol':1e-6, 'max_iter':10000, 'C':0.1,
        'ag_args': {'name_suffix': 'Lasso'}},
    {'multi_class':'multinomial', 'penalty':'elasticnet', 'tol':1e-6, 'max_iter':10000, 'C':0.1,
        'ag_args': {'name_suffix': 'ElasticNet'}}
    ]

custom_preset = {'auto_stack': True,
                'dynamic_stacking': False,
                'hyperparameters':custom_hyperparameters,
                'refit_full': False,
                'set_best_to_refit_full': False,
                'save_bag_folds': True,
                'time_limit': time_limit,
                'infer_limit': infer_limit}

# %%
np.random.seed(2024)

predictor = TabularPredictor(label=label, problem_type='multiclass', eval_metric='log_loss', log_to_file=True)

predictor.fit(final_train_data, presets=custom_preset)  

# %%
predictor.persist(models='all', max_memory=0.5) # improves prediction time, consumes more memory

df_leaders = predictor.leaderboard(final_test_data)

df_leaders.head(40)
# %%
df_leaders.tail(20)


# %%

y_pred = predictor.predict_proba(final_test_data.drop(columns=[label]), model=finalModel)

pd.concat([final_test_data[label], y_pred], axis=1).head(10)

# %%

df_important_features = predictor.feature_importance(final_test_data, model=finalModel)

df_important_features.reset_index()
# %%

final_low_importance_features = df_important_features[df_important_features['importance'] < 0.001].index.to_list()

final_low_importance_features
# %%
