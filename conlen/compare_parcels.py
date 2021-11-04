import numpy as np
import pandas as pd

fed = pd.read_csv('parcel_comparison_data/spm_ss_mROI_data.conlen2_fed.csv', sep=',\s?', encoding='utf-8')

pdd = pd.read_csv('parcel_comparison_data/spm_ss_mROI_data.conlen2_pdd.csv', sep=',\s?', encoding='utf-8')

fROIs = {
    1: 'LIFGorb',
    2: 'LIFG',
    3: 'LAntTemp',
    4: 'LPostTemp',
    5: 'LAngG'
}

for fROI in fROIs:
    _fed = fed[fed.ROI == fROI]
    _pdd = pdd[pdd.ROI == fROI]

    assert np.allclose((_fed[['ROI', 'Subject', 'Effect']] == _pdd[['ROI', 'Subject', 'Effect']]).mean(), 1), 'Misaligned rows'

    print('fROI: %s' % fROIs[fROI])
    print('r:    %0.2f' % np.round(np.corrcoef(_fed.EffectSize, _pdd.EffectSize)[0,1], 2))
    print('')