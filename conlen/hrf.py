import sys, argparse, pandas as pd, numpy as np
from mvpa2.misc.data_generators import double_gamma_hrf as hrf

def hrf_convolve(df_stim, df_res, grouping_columns):
    keys = []
    to_add = []
    for g in grouping_columns:
        if g in df_stim.columns:
            keys.append(g)
            df_stim[g] = df_stim[g].astype('category')
        else:
            to_add.append(g)

    cols = [x for x in df_stim.select_dtypes([np.number]).columns if x != 'time']

    stim_series = {x: y for x, y in df_stim.groupby(keys)}
    res_series = df_res.groupby(to_add + keys)

    out = []
    for key, res_cur in res_series:
        stim_key = key[len(to_add):]
        stim_cur = stim_series[stim_key]
        if 'duration' in stim_cur.columns:
            duration = stim_cur['duration']
        else:
            duration = None
        X = stim_cur[cols]
        impulse_times = stim_cur.time.values
        response_times = res_cur.time.values
        D = response_times[..., None] - impulse_times[None, ...]
        G_mask = D >= 0
        G = hrf(D)
        G = np.where(G_mask, G, 0)
        if duration is not None:
            X = X.multiply(duration, axis=0)
        X_conv = np.dot(G, X)
        X_conv = pd.DataFrame(X_conv, columns=cols)
        X_conv['time'] = response_times
        if 'tr' in res_cur:
            stim_cur['tr'] = res_cur['tr']
        if 'sampleid' in res_cur:
            stim_cur['sampleid'] = res_cur['sampleid']
        for col, val in zip(to_add + keys, key):
            X_conv[col] = val
        out.append(X_conv.reset_index(drop=True))

    out = pd.concat(out, axis=0)
    out.reset_index(drop=True, inplace=True)
    for c in df_res.columns:
        if c not in out.columns:
            out[c] = df_res[c]

    return out

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Convolve data table using HRF')
    argparser.add_argument('stimuli', type=str, help='Path to stimulus data')
    argparser.add_argument('response', type=str, help='Path to response data')
    argparser.add_argument('-g', '--grouping_columns', nargs='+', default=['docid'], help='Name(s) of column(s) that define(s) unique time series.')
    args = argparser.parse_args()

    df_stim = pd.read_csv(args.stimuli, sep=' ', skipinitialspace=True)
    df_stim['rate'] = 1.
    df_res = pd.read_csv(args.response, sep=' ', skipinitialspace=True)

    out = hrf_convolve(df_stim, df_res, args.grouping_columns)

    out.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')

