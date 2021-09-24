import sys
import re
import os
import json
import numpy as np
import pandas as pd
import scipy.io as io
from conlen.hrf import hrf_convolve

regnum = re.compile('(.+)_region([0-9]+)')
runnum = re.compile('run([0-9]+)')
clen = re.compile('[^0-9]*([0-9]+)')

def lookup_roi_name(s):
    kind, ix = regnum.match(s).groups()
    ix = int(ix) - 1
    if kind == 'language':
        prefix = 'boldLANG'
        names = language_roi
    elif kind == 'md':
        prefix = 'boldMD'
        names = md_roi
    else:
        raise ValueError('Unrecognized network "%s" in data column' % kind)

    return prefix + names[ix]

def parse_path(path, kind='fmri'):
    if kind.lower() == 'behavioral':
        p_ix = 3
        r_ix = 5
    elif kind.lower() == 'fmri':
        p_ix = 1
        r_ix = 2
    else:
        raise ValueError('Unrecognized path kind "%s".' % kind)

    path = path[:-4].strip().split('/')[-1]
    fields = path.split('_')
    subject = 's'+fields[p_ix]
    run = int(runnum.match(fields[r_ix]).group(1))

    return subject, run

def z_trans(x):
    mu = x.mean()
    sd = x.std()

    return (x - mu) / sd

def process_fmri_data(df, z=False):
    df = pd.DataFrame(df)
    if z:
        for col in df.columns:
            if regnum.match(col):
                df[col] = z_trans(df[col])

    df.rename(mapper=lookup_roi_name, axis=1, inplace=True)

    return df

def cond2type(c):
    if c.startswith('cond'):
        out = 'C'
    elif '_jab' in c:
        out = 'JAB'
    elif c.endswith('nc'):
        out = 'NC'
    elif c.endswith('c'):
        out = 'C'
    else:
        out = c

    return out

def cond2len(c):
    match = clen.match(c)
    if match:
        return int(match.group(1))
    return 0


def cond2code(t, l):
    if t == 'C':
        if l == 12:
            out = 'A'
        elif l == 6:
            out = 'B'
        elif l == 4:
            out = 'C'
        elif l == 3:
            out = 'E'
        elif l == 2:
            out = 'G'
        elif l == 1:
            out = 'H'
        else:
            raise ValueError('Unrecognized length %d for condition %s.' % (l, t))
    elif t == 'NC':
        if l == 4:
            out = 'D'
        elif l == 3:
            out = 'F'
        else:
            raise ValueError('Unrecognized length %d for condition %s.' % (l, t))
    elif t == 'JAB':
        if l == 12:
            out = 'I'
        elif l == 4:
            out = 'J'
        elif l == 1:
            out = 'K'
        else:
            raise ValueError('Unrecognized length %d for condition %s.' % (l, t))
    else:
        out = 'FIX'

    return out

def get_cond_exp1(x):
    out = x
    if x == 'cond12':
        out = 'A_12c'
    elif x == 'cond6':
        out = 'B_6c'
    elif x == 'cond4':
        out = 'C_4c'
    elif x == 'cond3':
        out = 'E_3c'
    elif x == 'cond2':
        out = 'G_2c'
    elif x == 'cond1':
        out = 'H_1c'

    return out

def get_docid(x):
    if x.condcode == 'FIX':
        return 'FIX'
    return x.condcode + str(x.itemnum)

hemispheres = ['L', 'R']

language_roi_src = [
    'IFGorb',
    'IFG',
    'MFG',
    'AntTemp',
    'PostTemp',
    'AngG'
]

md_roi_src = [
    'PostPar',
    'MidPar',
    'AntPar',
    'SFG',
    'PrecG',
    'IFGop',
    'MFG',
    'MFGorb',
    'Insula',
    'mPFC'
]

language_roi = ['L' + x for x in language_roi_src]
md_roi = [x + y for x in hemispheres for y in md_roi_src]

language_columns = ['boldLang' + h + x for h in hemispheres for x in language_roi]
md_columns = ['boldMD' + h + x for h in hemispheres for x in md_roi]

colmap = {
    'Condition': 'cond',
    'Item': 'item',
    'ItemNum': 'itemnum',
    'Onsets': 'onset'
}

cl_predictor_path = 'data/conlen/ling_preds/'

# PMI
df_pmi = pd.read_csv(cl_predictor_path + '/pPMI_incremental_long.csv')
df_pmi['cond'] = df_pmi.Condition
del df_pmi['Condition']
df_pmi['conlen'] = df_pmi.cond.map(cond2len)
df_pmi['condstr'] = df_pmi.cond
df_pmi.cond = df_pmi.cond.map(cond2type)
df_pmi['condcode'] = df_pmi[['cond', 'conlen']].apply(lambda x: cond2code(*x), axis=1)
df_pmi['sentid'] = df_pmi.ItemNum - 1
df_pmi['itemnum'] = df_pmi['ItemNum'].astype(str)
del df_pmi['ItemNum']
df_pmi['sentpos'] = df_pmi.Position
del df_pmi['Position']
df_pmi['docid'] = df_pmi.apply(get_docid, axis=1)
df_pmi['word'] = df_pmi.Word.str.lower()
del df_pmi['Word']

# LING

ling_preds = [
    'word',
    'wlen',
    'docid',
    'sentid',
    'sentpos',
    'cond',
    'condcode',
    'conlen',
    'dlt',
    'dltc',
    'dltv',
    'dltm',
    'dltcv',
    'dltcm',
    'dltvm',
    'dltcvm',
    'dlts',
    'unigramsurp',
    'fwprob5surp',
    'totsurp',
    'startembdAny',
    'endembdAny',
    'embddepthAny',
    'embdlen',
    'noF',
    'noFlen',
    'noFlenlog1p',
    'noFdr',
    'noFdrv',
    'yesJ'
]

ling_preds_full = ling_preds + ['PMI']

ling_preds_nojab = [
    'dlt',
    'dltc',
    'dltv',
    'dltm',
    'dltcv',
    'dltcm',
    'dltvm',
    'dltcvm',
    'dlts',
    'unigramsurp',
    'fwprob5surp',
    'totsurp',
    'startembdAny',
    'endembdAny',
    'embddepthAny',
    'embdlen',
    'noF',
    'noFlen',
    'noFlenlog1p',
    'noFdr',
    'noFdrv',
    'yesJ'
]

conlen_itemmeasures = pd.read_csv(
    cl_predictor_path + '/conlen2fmri.wsj02to21-gcg15-nol-prtrm-3sm-synproc-+c_+u_+b5000_parsed.dlt.lc.unigram.5-kenlm.all-itemmeasures',
    sep=' '
)
conlen_itemmeasures['noFlenlog1p'] = np.log(conlen_itemmeasures['noFlen'].values + 1)
conlen_itemmeasures = conlen_itemmeasures[ling_preds]
conlen_itemmeasures.loc[conlen_itemmeasures.cond == 'JAB', ling_preds_nojab] = 0
conlen_itemmeasures = pd.merge(conlen_itemmeasures, df_pmi[['PMI', 'docid', 'sentpos', 'cond', 'conlen', 'condcode']],
                               how='left', on=['docid', 'sentpos', 'cond', 'conlen', 'condcode'])
conlen_itemmeasures = conlen_itemmeasures.fillna(0.)
conlen_itemmeasures['itempos'] = conlen_itemmeasures.groupby('docid').cumcount() + 1

means = conlen_itemmeasures[ling_preds_nojab + ['docid']].groupby(['docid']).mean().reset_index()
means.columns = [x if x == 'docid' else x + 'Mean' for x in means.columns]
conlen_itemmeasures = pd.merge(conlen_itemmeasures, means, on='docid', how='left')

for exp in ['1', '2']:
    cl_beh_path = 'data/conlen/Nlength_con%s/' % exp
    cl_fmri_path = 'data/conlen/results/Nlength_con%s/' % exp
    paths = [cl_fmri_path + x for x in os.listdir(cl_fmri_path) if (x.endswith('mat') and not 'sess2' in x)]
    path_map = 'data/conlen/nlength_con%s_behavioral.csv' % exp

    path_map = pd.read_csv(path_map, header=None)
    path_map = dict(zip(
        path_map[0],
        path_map[1]
    ))

    runsets = []
    timeseries = []
    stim = []

    stim_length = {}

    lengths_c = [1, 2, 3, 4, 6, 12]
    lengths_nc = [3, 4]
    lengths_jab = [1, 4, 12]

    with open('failures_conlen%s.txt' % exp, 'w') as fail:
        for path in paths:
            sys.stderr.write('Processing data file %s...\n' % path)
            sys.stderr.flush()
            subject, run = parse_path(path, kind='fmri')
            df_response = io.loadmat(path, matlab_compatible=True)['time_courses']
            df_response = {df_response.dtype.names[i]: df_response[0][0][i].mean(axis=0) for i in range(len(df_response[0][0]))}

            if path.split('/')[-1] in path_map:
                beh_path = cl_beh_path + path_map[path.split('/')[-1]]

                if exp == '1':
                    columns = ['Onsets', 'Condition', 'ItemNum', 'Item']
                else:
                    columns = ['Onsets', 'Condition', 'unknown', 'ItemNum', 'Item']

                df_stim = io.loadmat(beh_path, matlab_compatible=True)['behmat']
                df_stim = [
                    [''.join([str(x1) for x2 in x3 for x1 in x2]) for x3 in x4] for x4 in df_stim
                ]
                df_stim = pd.DataFrame(df_stim, columns=columns)
                df_stim.Onsets = df_stim.Onsets.astype(float)
                df_stim.ItemNum = df_stim.ItemNum.str.replace('\.0', '')
                if exp == '1':
                    df_stim.Condition = df_stim.Condition.apply(get_cond_exp1)
                    df_stim.Item = df_stim.Item.apply(lambda x: 'NA' if x == 'NULL' else x)
                else: # exp == '2'
                    del df_stim['unknown']
                df_stim.rename(mapper=colmap, axis=1, inplace=True)

                df_response = process_fmri_data(df_response)
                df_response['subject'] = subject
                df_response['run'] = run
                df_response['experiment'] = 'CON' + exp
                arr = np.arange(len(df_response))
                df_response['tr'] = arr + 1
                df_response['time'] = arr * 2
                df_response['trrev'] = np.arange(len(df_response), 0, -1)
                df_response['sampleid'] = df_response['subject'].astype(str).str.cat(
                    df_response['run'].astype(str).str.cat(
                        df_response['tr'].apply('{0:05d}'.format),
                        sep='-'
                    ),
                    sep='-'
                )
                df_response.reset_index(drop=True, inplace=True)
                df_stim['subject'] = subject
                df_stim['run'] = run
                df_stim['experiment'] = 'CON' + exp
                df_stim['isFIX'] = (df_stim.itemnum == 'NA').astype(int)

                df_fix = df_stim[df_stim.isFIX == 0]
                df_fix.isFIX = 1
                df_fix.onset += 3.6

                df_stim = pd.concat([df_stim, df_fix], axis=0)
                df_stim = df_stim.sort_values('onset').reset_index(drop=True)

                sel = df_stim.isFIX == 1

                df_stim.loc[sel, 'cond'] = 'FIX'
                df_stim.loc[sel, 'itemnum'] = 0
                df_stim.loc[sel, 'item'] = 'FIX'
                df_stim.loc[sel, 'isFIX'] = 1
                df_stim.loc[sel, 'condcode'] = 'FIX'
                df_stim.loc[sel, 'conlen'] = 0
                df_stim.loc[sel, 'docid'] = 'FIX'

                df_stim['sessionpos'] = np.arange(len(df_stim))
                df_stim['offset'] = df_stim.onset.shift(-1)
                df_stim['conlen'] = df_stim.cond.map(cond2len)
                for length in lengths_c:
                    df_stim['isLen%d' % length] = (df_stim.conlen == length).astype(int)
                df_stim['condstr'] = df_stim.cond
                df_stim.cond = df_stim.cond.map(cond2type)
                df_stim['condcode'] = df_stim[['cond', 'conlen']].apply(lambda x: cond2code(*x), axis=1)
                df_stim['docid'] = df_stim.apply(get_docid, axis=1)
                df_stim = df_stim[~pd.isnull(df_stim.offset)]

                runset = frozenset(df_stim[~(df_stim.isFIX == 1)].docid)
                if runset not in runsets:
                    runsets.append(runset)
                df_stim['runset'] = runsets.index(runset)

                words = df_stim['item'].str.split(' ').apply(pd.Series, 1).stack().str.lower()
                words.index = words.index.droplevel(-1)
                words.name = 'word'
                df_stim = df_stim.join(words)
                df_stim.reset_index(inplace=True, drop=True)
                df_stim['itempos'] = df_stim.groupby('docid').cumcount() + 1
                df_stim['isEND'] = (df_stim.itempos % df_stim.conlen == 0).astype(int)
                del df_stim['item']
                del df_stim['word']

                duration = df_stim.offset - df_stim.onset
                fixn_loc = df_stim.isFIX
                reps = np.where(fixn_loc, duration // 0.3, np.ones(len(df_stim))).astype(int)
                cols = df_stim.columns
                dtypes = df_stim.dtypes
                expanded = np.repeat(df_stim.values, reps, axis=0)
                df_stim = pd.DataFrame(expanded, columns=cols)
                for i, x in enumerate(df_stim.columns):
                    df_stim[x] = df_stim[x].astype(dtypes[i])

                on = ['docid', 'cond', 'conlen', 'condcode', 'itempos']
                df_stim = pd.merge(df_stim, conlen_itemmeasures, on=on, how='left')
                df_stim.loc[df_stim.isFIX == 1, 'word'] = 'FIX'
                df_stim['time'] = df_stim.onset + df_stim.groupby(['sessionpos']).cumcount() * 0.3
                df_stim = df_stim.fillna(0.)
                df_stim['duration'] = 0.3
                df_stim.reset_index(drop=True, inplace=True)

                timeseries.append(df_response)
                stim.append(df_stim)
            else:
                fail.write(path + '\n')

    # Response
    timeseries = pd.concat(timeseries, axis=0)
    timeseries.reset_index(drop=True, inplace=True)

    # Predictors
    stim = pd.concat(stim, axis=0)
    stim.reset_index(drop=True, inplace=True)
    stim['isC'] = (stim.cond == 'C').astype(int)
    stim['isNC'] = (stim.cond == 'NC').astype(int)
    stim['isJAB'] = (stim.cond == 'JAB').astype(int)
    for length in lengths_c:
        stim['isCLen%d' % length] = stim['isC'] * stim['isLen%d' % length]
        for ling_pred in ling_preds_full:
            stim['isCLen%d%s' % (length, ling_pred)] = stim['isCLen%d' % length] * stim[ling_pred]
    for length in lengths_nc:
        stim['isNCLen%d' % length] = stim['isNC'] * stim['isLen%d' % length]
        for ling_pred in ling_preds_full:
            stim['isNCLen%d%s' % (length, ling_pred)] = stim['isNCLen%d' % length] * stim[ling_pred]
    for length in lengths_jab:
        stim['isJABLen%d' % length] = stim['isJAB'] * stim['isLen%d' % length]
        for ling_pred in ling_preds_full:
            stim['isJABLen%d%s' % (length, ling_pred)] = stim['isJABLen%d' % length] * stim[ling_pred]
    stim['isCLen'] = stim['isC'] * stim['conlen']

    loglen = np.where(stim['conlen'] > 0, np.log(stim['conlen']), 0)

    stim['isCLogLen'] = stim['isC'] * loglen
    stim['isNCLen'] = stim['isNC'] * stim['conlen']
    stim['isNCLogLen'] = stim['isNC'] * loglen
    stim['isJABLen'] = stim['isJAB'] * stim['conlen']
    stim['isJABLogLen'] = stim['isJAB'] * loglen

    if not os.path.exists('output/conlen/nlength_con%s' % exp):
        os.makedirs('output/conlen/nlength_con%s' % exp)

    print('Saving wide data...')

    conlen_itemmeasures.to_csv('output/conlen/nlength_con%s/conlen%s.itemmeasures' % (exp, exp), sep=' ', na_rep='NaN', index=False)
    # timeseries.to_csv('output/conlen/nlength_con%s/conlen%sfmri_bold_wide.csv' % (exp, exp), sep=' ', na_rep='NaN', index=False)
    # stim.to_csv('output/conlen/nlength_con%s/conlen%sfmri_stim_wide.csv' % (exp, exp), sep=' ', na_rep='NaN', index=False)

    stim_key_cols = ['subject', 'run', 'experiment']
    res_key_cols = ['subject', 'run', 'experiment', 'fROI']

    print('Convolving...')

    stim_hrf = hrf_convolve(stim[[x for x in stim.columns if not x.startswith('bold')]], timeseries, stim_key_cols)
    stim_hrf = stim_hrf[[x for x in stim_hrf.columns if not x.startswith('bold')]]

    # print('Saving response data...')

    timeseries_long = pd.melt(
        timeseries,
        var_name='fROI',
        id_vars=[x for x in timeseries.columns if not x.startswith('bold')],
        value_vars=[x for x in timeseries.columns if x.startswith('bold')],
        value_name='BOLD'
    )
    timeseries_long.fROI = timeseries_long.fROI.apply(lambda x: x[4:])
    # timeseries_long.to_csv('output/conlen/nlength_con%s/conlen%sfmri_bold_long.csv' % (exp, exp), sep=' ', na_rep='NaN', index=False)

    print('Computing long stim data...')

    hrf_dict = {x[0]:x[1] for x in stim_hrf.groupby(stim_key_cols)}
    stim_dict = {x[0]:x[1] for x in stim.groupby(stim_key_cols)}
    res_keys = timeseries_long[res_key_cols].drop_duplicates().values.tolist()
    res_dict = {x[0]:x[1] for x in timeseries_long.groupby(res_key_cols)}
    hrf_long = []
    stim_long = []
    for x in res_keys:
        stim_key = tuple(x[:-1])

        hrf_long_cur = hrf_dict[stim_key].copy().reset_index(drop=True)
        hrf_long_cur['fROI'] = x[-1]
        BOLD = res_dict[tuple(x)].copy().reset_index(drop=True).BOLD
        hrf_long_cur['BOLD'] = BOLD
        hrf_long.append(hrf_long_cur)

        stim_long_cur = stim_dict[stim_key].copy()
        stim_long_cur['fROI'] = x[-1]
        stim_long.append(stim_long_cur)

    # print('Saving long HRF data...')

    hrf_long = pd.concat(hrf_long, axis=0)
    # hrf_long.to_csv('output/conlen/nlength_con%s/conlen%sfmri_hrf_long.csv' % (exp, exp), sep=' ', na_rep='NaN', index=False)

    # print('Saving long stim data...')

    # stim_long = pd.concat(stim_long, axis=0)
    # stim_long.to_csv('output/conlen/nlength_con%s/conlen%sfmri_stim_long.csv' % (exp, exp), sep=' ', na_rep='NaN', index=False)

    print('Saving long LANG data...')

    # timeseries_long_lang = timeseries_long[timeseries_long.fROI.apply(lambda x: x[4:]).isin(language_roi)]
    # stim_long_lang = stim_long[stim_long.fROI.apply(lambda x: x[4:]).isin(language_roi)]
    hrf_long_lang = hrf_long[hrf_long.fROI.apply(lambda x: x[4:]).isin(language_roi)]

    # timeseries_long_lang.to_csv('output/conlen/nlength_con%s/conlen%sfmri_bold_long_lang.csv' % (exp, exp), sep=' ', na_rep='NaN', index=False)
    # stim_long_lang.to_csv('output/conlen/nlength_con%s/conlen%sfmri_stim_long_lang.csv' % (exp, exp), sep=' ', na_rep='NaN', index=False)
    hrf_long_lang.to_csv('output/conlen/nlength_con%s/conlen%sfmri_hrf_long_lang.csv' % (exp, exp), sep=' ', na_rep='NaN', index=False)









