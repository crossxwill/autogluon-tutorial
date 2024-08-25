# %%
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt

df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/train.csv")
df.head()

# %%

train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
)
train_data.head()

# %%
predictor = TimeSeriesPredictor(
    prediction_length=48,
    target="target",
    eval_metric="MASE",
)

predictor.fit(
    train_data,
    time_limit=3600,
)
# %%
test_data = TimeSeriesDataFrame.from_path("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly_subset/test.csv")

test_data.head()

# %%
predictor.leaderboard(test_data)

# %%
finalModel = 'WeightedEnsemble'

predictions = predictor.predict(train_data, model=finalModel)
predictions.head()

# %%

predictor.plot(test_data, predictions, quantile_levels=[0.1, 0.9], max_history_length=200, max_num_item_ids=4);

plt.show()
# %%
