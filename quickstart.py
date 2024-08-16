# %%
from autogluon.tabular import TabularDataset, TabularPredictor
# from autogluon.core.models import AbstractModel
# from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

label = 'signature'

data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'

train_data = TabularDataset(f'{data_url}train.csv')
test_data = TabularDataset(f'{data_url}test.csv')

train_data.head()

# %%
custom_preset = {'auto_stack': False, 'dynamic_stacking': False,
                'hyperparameters':'zeroshot', 'refit_full': True,
                'set_best_to_refit_full': True, 'save_bag_folds': False}

predictor = TabularPredictor(label=label, problem_type='multiclass', eval_metric='log_loss')

predictor.fit(train_data, holdout_frac=0.3, presets=custom_preset, included_model_types=['CAT', 'XGB'])

# %%
predictor.leaderboard(test_data)

# %%
y_pred = predictor.predict_proba(test_data.drop(columns=[label]))
y_pred.head()

# predictor.evaluate(test_data, silent=True)

# %%
