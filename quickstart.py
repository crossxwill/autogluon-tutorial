# %%
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

import numpy as np

from sklearnex import patch_sklearn
patch_sklearn()

label = 'signature'

data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'

train_data = TabularDataset(f'{data_url}train.csv')
test_data = TabularDataset(f'{data_url}test.csv')

train_data.head()

# %%
custom_hyperparameters = get_hyperparameter_config('zeroshot')
custom_hyperparameters['GBM'] = [{'objective':'multiclass', 'boosting_type':'dart', 'num_boost_round':5000}]
custom_hyperparameters['CAT'] = [{'loss_function':'Logloss', 'iterations':5000}]
custom_hyperparameters['RF'] = [{'criterion': 'log_loss', 'n_estimators':5000,'ag_args': {'name_suffix': 'LogLoss', 'problem_types': ['multiclass']}}]
custom_hyperparameters['XT'] = [{'criterion': 'log_loss', 'n_estimators':5000,'ag_args': {'name_suffix': 'LogLoss', 'problem_types': ['multiclass']}}]
custom_hyperparameters['LR'] = [{'multi_class':'multinomial', 'penalty':None, 'tol':1e-6, 'max_iter':10000}]

custom_preset = {'auto_stack': False, 'dynamic_stacking': False,
                'hyperparameters':custom_hyperparameters, 'refit_full': True,
                'set_best_to_refit_full': True, 'save_bag_folds': False}

# %%
np.random.seed(2024)

predictor = TabularPredictor(label=label, problem_type='multiclass', eval_metric='log_loss', log_to_file=True)

predictor.fit(train_data, holdout_frac=0.3, presets=custom_preset)

# %%
df_leaders = predictor.leaderboard(test_data)

df_leaders.head(40)
# %%
y_pred = predictor.predict_proba(test_data.drop(columns=[label]))
y_pred.head()

# predictor.evaluate(test_data, silent=True)

# %%
