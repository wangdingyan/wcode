import pandas as pd
from wcode.database.uniprot import id2seq
from copy import deepcopy
# df = pd.read_excel('D:\\nutshell\\Official\\GG\\A-江何伟老师项目\\20241111_LGTHS.xlsx')
df = pd.read_excel('D:\\nutshell\\Official\\GG\\A-江何伟老师项目\\20241111_LGTHS_XJ.xlsx')
# df['Protein'] = None
print(df)
df_out = deepcopy(df)

for i, p in enumerate(df['UniProt'].tolist()):
    if i <= 45:
        continue
    seq = id2seq(p)
    print(i, p, seq)
    df_out.iloc[i, 2] = seq
    print(df_out)
    df_out.to_excel('D:\\nutshell\\Official\\GG\\A-江何伟老师项目\\20241111_XJ2.xlsx', index=False)

