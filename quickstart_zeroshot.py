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

custom_preset = {'auto_stack': False, 'dynamic_stacking': False,
                'hyperparameters':custom_hyperparameters, 'refit_full': False,
                'set_best_to_refit_full': False, 'save_bag_folds': False}

# %%
np.random.seed(2024)

predictor = TabularPredictor(label=label, problem_type='multiclass', eval_metric='log_loss', log_to_file=True)

predictor.fit(train_data, presets=custom_preset, excluded_model_types=['KNN'])  

# %%
metrics = ['model', 'score_test', 'score_val', 'eval_metric', 'pred_time_test', 'fit_time']

df_leaders = predictor.leaderboard(test_data)

df_leaders.head(40)[metrics]
# %%
df_leaders.tail(20)[metrics]


# %%
y_pred = predictor.predict_proba(test_data.drop(columns=[label]))
y_pred.head()

# predictor.evaluate(test_data, silent=True)

# %%
# "AutogluonModels\ag-20240821_072221"