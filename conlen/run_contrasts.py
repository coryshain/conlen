import pickle
import os
import numpy as np
import pandas as pd
import argparse


def var(m, weights):
    w = np.array(weights)
    while len(w.shape) < 2:
        w = w[..., None]
    cov = m.cov_params()
    v = np.dot(w.T, np.dot(cov, w))
    v = np.squeeze(v)

    return v


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Compute contrasts for constituent length experiments
    ''')
    args = argparser.parse_args()

    for experiment in ['1', '2']:
        if experiment == '1':
            contrast_names = [
                'isCLen1',
                'isCLen2',
                'isCLen4',
                'isCLen6',
                'isCLen12',
                'CLen',
                'other'
            ]
            contrast_weights = {
                #               int     C1      C2      C4      C6    C12
                'isCLen1':     [  0,     1,      0,      0,      0,     0],
                'isCLen2':     [  0,     0,      1,      0,      0,     0],
                'isCLen4':     [  0,     0,      0,      1,      0,     0],
                'isCLen6': [0, 0, 0, 0, 1, 0],
                'isCLen12': [0, 0, 0, 0, 0, 1],
                'CLen': [0, -5.625, -3.375, 1, 3, 5],
                'other': [0, 0, 0, 0, 0, 0, 1],
            }
        else:  # experiment == '2'
            contrast_names = [
                'isCLen1',
                'isCLen2',
                'isCLen3',
                'isCLen4',
                'isCLen6',
                'isCLen12',
                'isNCLen3',
                'isNCLen4',
                'isJABLen1',
                'isJABLen4',
                'isJABLen12',
                'isC',
                'isC34',
                'isC1412',
                'isNC',
                'isJAB',
                'CLen',
                'CLen34',
                'CLen1412',
                'NCLen',
                'JABLen',
                'C>JAB',
                'C>NC',
                'CLen>JABLen',
                'CLen>NCLen',
                'other'
            ]
            contrast_weights = {
                #               int     C1      C2      C3      C4      C6      C12     NC3     NC4     J1      J4    J12
                'isCLen1': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'isCLen2': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'isCLen3': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                'isCLen4': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                'isCLen6': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                'isCLen12': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                'isNCLen3': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                'isNCLen4': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'isJABLen1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                'isJABLen4': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                'isJABLen12': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                'isC': [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                'isC34': [0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0],
                'isC1412': [0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
                'isNC': [0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0],
                'isJAB': [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
                'CLen': [0, -5, -3, -1, 1, 3, 5, 0, 0, 0, 0, 0],
                'CLen34': [0, 0, 0, -9, 9, 0, 0, 0, 0, 0, 0, 0],
                'CLen1412': [0, -9, 0, 0, 1.5, 0, 7.5, 0, 0, 0, 0, 0],
                'NCLen': [0, 0, 0, 0, 0, 0, 0, -9, 9, 0, 0, 0],
                'JABLen': [0, 0, 0, 0, 0, 0, 0, 0, 0, -9, 1.5, 7.5],
                'C>JAB': [0, 2, 0, 2, 2, 0, 0, 0, 0, -2, -2, -2],
                'C>NC': [0, 0, 0, 3, 3, 0, 0, -3, -3, 0, 0, 0],
                'CLen>JABLen': [0, -9, 0, 0, 1.5, 0, 7.5, 0, 0, 9, -1.5, -7.5],
                'CLen>NCLen': [0, 0, 0, -9, 9, 0, 0, 9, -9, 0, 0, 0],
                'other': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            }

        prefix = 'output/conlen/nlength_con%s/' % experiment

        contrasts_main = {}

        for path in [prefix + 'glm/' + x for x in os.listdir(prefix + '/glm/') if x.endswith('.obj')]:
            baseline_key = path.split('.')[-2]
            if baseline_key:
                baseline_key_str = baseline_key
            else:
                baseline_key_str = 'none'
            with open(path, 'rb') as f:
                models = pickle.load(f)

            contrasts = []

            for k in models:
                if len(k) == 2:
                    subject, froi = k
                    ling_pred = 'none'
                else:
                    subject, froi, ling_pred = k
                contrasts_cur = []
                contrast_diffs_cur = []
                names = contrast_names[:-1]
                if baseline_key_str != 'none':
                    names.append(baseline_key_str)

                for name in names:

                    b = models[k].params
                    if name in contrast_weights:
                        w = contrast_weights[name]
                    else:
                        w = contrast_weights['other']
                    w = np.array(w)
                    w = np.pad(w, [(0, len(b) - len(w))], mode='constant')[..., None]
                    m = np.squeeze(np.dot(b, w))
                    s = np.sqrt(var(models[k], w))
                    contrasts_cur.append((subject, froi, ling_pred, name, m, s, m / s))
                    if baseline_key_str == 'none':
                        if k not in contrasts_main:
                            contrasts_main[k] = {}
                        contrasts_main[k][name] = m
                    elif name != baseline_key_str:
                        contrast_diffs_cur.append((subject, froi, ling_pred, name + 'Diff', m - contrasts_main[k][name], np.nan, np.nan))

                contrasts_cur = pd.DataFrame(contrasts_cur + contrast_diffs_cur, columns=['subject', 'fROI', 'ling', 'contrast', 'estimate', 'se', 't'])

                contrasts.append(contrasts_cur)

            contrasts = pd.concat(contrasts, axis=0)

            if not os.path.exists(prefix + 'contrasts/'):
                os.makedirs(prefix + 'contrasts/')

            contrasts.to_csv(prefix + 'contrasts/contrasts.%s.csv' % baseline_key_str, index=False)


