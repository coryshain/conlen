import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

df1 = pd.read_csv('fed10_data/spm_ss_mROI_data.exp1.csv', sep=',\s?')
df1 = df1[df1.Effect == 'S-N']
subj1 = {x[:3] for x in df1.Subject.unique()}

df2 = pd.read_csv('fed10_data/spm_ss_mROI_data.exp2.csv', sep=',\s?')
df2 = df2[df2.Effect == 'S-N']
subj2 = {x[:3] for x in df2.Subject.unique()}

fROIs = {
    1: 'LIFGorb',
    2: 'LIFG',
    3: 'LMFG',
    4: 'LAntTemp',
    5: 'LPostTemp',
    6: 'LAngG'
}

out = []

for i in range(1, 7):
    contrasts = np.concatenate([
        df1[df1.ROI == i].EffectSize.values,
        df2[df2.ROI == i].EffectSize.values
    ])

    t, p = ttest_1samp(contrasts, 0.)
    d = contrasts.mean() / contrasts.std(ddof=1)
    out.append((fROIs[i], t, p, d))

out = pd.DataFrame(out, columns=['fROI', 't', 'p', 'd'])
out['p_fdr'] = fdrcorrection(out.p, method='negcorr')[1]
out.to_csv('output/conlen/localizer_stats.csv')

