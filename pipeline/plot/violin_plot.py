import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns




df = pd.read_csv('/mnt/c/tmp/test_set_prediction_round2.csv')

df['test_predictions'] = np.log10(df['test_predictions']+1e-30)
positive_prediction = []
negative_prediction = []

for item in df.iterrows():
    if item[1]['test_labels'] == 1:
        positive_prediction.append(np.log(item[1]['test_predictions']))
    else:
        negative_prediction.append(np.log(item[1]['test_predictions']))

fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
sns.violinplot(df, x='test_labels', y='test_predictions', inner_kws=dict(box_width=15, whis_width=2, color=".8"))
plt.show()