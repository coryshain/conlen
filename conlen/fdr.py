import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    FDR correct for comparisons in ConLen study
    ''')
    args = argparser.parse_args()

    path = 'output/conlen/signif.csv'

    df = pd.read_csv(path)

    fROIs = list(df.fROI.unique())

    # Main result

    contrasts = ['CLen', 'JABLen', 'C_gt_JAB', 'CLen_gt_JABLen', 'CLen34', 'NCLen', 'C_gt_NC', 'CLen_gt_NCLen']

    main = []

    for exp in (1, 2):
        for contrast in contrasts:
            _df = df[(df.experiment == exp) & (df.baseline == 'none') & (df.contrast == contrast)]
            p = _df.p.values
            reject, p = fdrcorrection(p, method='negcorr')
            _df['p_fdr'] = p
            _df['p_fdr_reject'] = reject

            main.append(_df)

    main = pd.concat(main)
    main.to_csv('output/conlen/signif_main.csv', index=False)

    # Linguistic re-analysis

    ling = []

    ling_vars = set(df.baseline.unique()) - {'none'}

    for exp in (1, 2):
        for ling_var in ling_vars:
            _df = df[(df.experiment == exp) & (df.baseline == ling_var) & (df.contrast == 'CLenDiff')]
            p = _df.p.values
            reject, p = fdrcorrection(p, method='negcorr')
            _df['p_fdr'] = p
            _df['p_fdr_reject'] = reject

            ling.append(_df)

    ling = pd.concat(ling)
    ling.to_csv('output/conlen/signif_ling_diff.csv', index=False)