import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


df_train = pd.read_csv('../labels.csv')

target_count = df_train.label.value_counts()

y = df_train['label']
X = df_train.drop(['label'], axis=1)
rus = RandomUnderSampler()

X_rus, y_rus = rus.fit_sample(X, y)


print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

