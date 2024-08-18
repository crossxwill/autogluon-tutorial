
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

# data
label = 'signature'
data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
train_data = TabularDataset(f'{data_url}train.csv')
test_data = TabularDataset(f'{data_url}test.csv')

# train
np.random.seed(2024)
predictor = TabularPredictor(label=label, problem_type='multiclass', eval_metric='log_loss', log_to_file=True)
predictor.fit(train_data, included_model_types=['XGB', 'CAT'])  

# report
metrics = ['model', 'score_test', 'score_val', 'eval_metric', 'pred_time_test', 'fit_time']
predictor.leaderboard(test_data)[metrics]

