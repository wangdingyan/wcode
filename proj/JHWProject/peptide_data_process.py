import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.impute import KNNImputer
from collections import Counter

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv('/mnt/c/tmp/YS1917_25-32_Report_peptide-Report-short (Pivot).tsv', sep='\t')

column_names = df.columns
seq_column = ['PEP.StrippedSequence']
quantity_column_names = [n for n in column_names if 'PEP.Quantity' in n]
exp_1_names = quantity_column_names[:3]
exp_2_names = quantity_column_names[3:6]
ref_names = quantity_column_names[6:]
seq_column.extend(quantity_column_names)
df = df[seq_column]
df = df.drop_duplicates(subset='PEP.StrippedSequence', keep='first').reset_index(drop=True)
print("************************")
print(df[:10])
count_replaced = (df == 1.0).sum().sum()
df = df.replace(1, np.nan)
print("************************")
print(df[:10])
# print(df_unique[:5])
drop_index = []

for l in tqdm(df.iterrows()):
    index = l[0]
    exp_1_none_num = l[1][exp_1_names].isnull().values.sum()
    exp_2_none_num = l[1][exp_2_names].isnull().values.sum()
    ref_none_num = l[1][ref_names].isnull().values.sum()
    if exp_1_none_num == 2 or exp_2_none_num == 2 or ref_none_num == 1:
        drop_index.append(index)

    if exp_1_none_num == 1:
        impute_value = np.nanmean((l[1][exp_1_names]).values)
        sel_idx = l[1][exp_1_names].isnull().argmax()
        sel_name = exp_1_names[sel_idx]
        df.loc[index, sel_name] = impute_value

    if exp_2_none_num == 1:
        impute_value = np.nanmean((l[1][exp_2_names]).values)
        sel_idx = l[1][exp_2_names].isnull().argmax()
        sel_name = exp_2_names[sel_idx]
        df.loc[index, sel_name] = impute_value


print("************************")
print(df[:10])

df = df.drop(drop_index)
print("************************")
print(df[:10])

values = df[quantity_column_names].values
imputer = KNNImputer(n_neighbors=5)
new_values = imputer.fit_transform(values)
df[quantity_column_names] = new_values

test_set = []
for l in tqdm(df.iterrows()):
    index = l[0]
    exp_1_none_num = l[1][exp_1_names].isnull().values.sum()
    exp_2_none_num = l[1][exp_2_names].isnull().values.sum()
    test_set.append(exp_1_none_num)
    test_set.append(exp_2_none_num)

print(Counter(test_set))
print("************************")
print(df[:10])

has_nan = df.isnull().values.any()
print(f"DataFrame contains NaN values: {has_nan}")

# 将 DataFrame 导出为 .tsv 文件
df.to_csv('/mnt/c/tmp/WDY_YS1917_25-32_Report_peptide-Report-short (Pivot).tsv', sep='\t', index=False)




# min_value = df_unique[quantity_column_names].min().min()
# min_row = df_unique[df_unique.isin([min_value]).any(axis=1)]
# pep_value = min_row['PEP.StrippedSequence']
# print(f"最小值是: {min_value}, 对应的 'PEP.StrippedSequence' 列的值是: {pep_value}")