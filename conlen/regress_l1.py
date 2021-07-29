import pickle
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import argparse

def z(x):
    return (x - x.mean()) / x.std()

def var(m, weights):
    w = np.array(weights)
    while len(w.shape) < 2:
        w = w[..., None]
    cov = m.cov_params()

    return np.dot(w.T, np.dot(cov, w))

ling_baselines = [
    [],
    ['wlen'],
    ['unigramsurp'],
    ['fwprob5surp'],
    ['totsurp'],
    ['dltcvm'],
    ['dlts'],
    ['PMI']
]

cols = {
    '1': [
        'isCLen1%s',
        'isCLen2%s',
        'isCLen4%s',
        'isCLen6%s',
        'isCLen12%s'
    ],
    '2': [
        'isCLen1%s',
        'isCLen2%s',
        'isCLen3%s',
        'isCLen4%s',
        'isCLen6%s',
        'isCLen12%s',
        'isNCLen3%s',
        'isNCLen4%s',
        'isJABLen1%s',
        'isJABLen4%s',
        'isJABLen12%s'
    ]
}

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Regress constituent length experiments
    ''')
    args = argparser.parse_args()

    for experiment in ['1', '2']:
        print('Reading experiment %s data...' % experiment)

        df = pd.read_csv('output/conlen/nlength_con%s/conlen%sfmri_hrf_long_lang.csv' % (experiment, experiment), sep=' ')

        print('Fitting GLM...')
        for baseline in ling_baselines:
            baseline_key = '_'.join(baseline)
            models = {}
            for k, v in df.groupby(['subject', 'fROI']):
                # tr_min = 6
                tr_min = 1
                if experiment == '1':
                    # tr_max = 103
                    tr_max = 108
                else: # experiment == '2'
                    # tr_max = 115
                    tr_max = 120
                v = v[(v.tr >= tr_min) & (v.tr <= tr_max)]
                y = v['BOLD']
                if len(y[~y.isna()]):
                    _cols = [x % '' for x in cols[experiment]]
                    X = sm.add_constant(v[_cols + baseline])
                    X = X.rename(lambda x: 'Intercept' if x == 'const' else x, axis=1)
                    for pred in baseline:
                        X[pred] = z(X[pred])
                    m = sm.OLS(y, X)
                    models[k] = m.fit()
                    if not baseline:
                        for ling_preds in ling_baselines:
                            for ling_pred in ling_preds:
                                _k = k + (ling_pred,)
                                _cols = [x % ling_pred for x in cols[experiment]] + [x % '' for x in cols[experiment]]
                                X = sm.add_constant(v[_cols])
                                X = X.rename(lambda x: 'Intercept' if x == 'const' else x, axis=1)
                                m = sm.OLS(y, X)
                                models[_k] = m.fit()
                else:
                    print('Model %s had all NaN response measures. Skipping...\n' % ', '.join(k))

            if not os.path.exists('output/conlen/nlength_con%s/glm' % experiment):
                os.makedirs('output/conlen/nlength_con%s/glm' % experiment)

            with open('output/conlen/nlength_con%s/glm/glm.%s.obj' % (experiment, baseline_key), 'wb') as f:
                pickle.dump(models, f)
