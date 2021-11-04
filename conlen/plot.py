import os
import re
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import Divider, Size
from statsmodels.nonparametric.smoothers_lowess import lowess
import argparse

def get_color(name, color):
    if name.endswith('surprisal'):
        return tuple([x for x in color[:3]] + [0.4])
    if name == 'DLT':
        return color
    return tuple([x for x in color[:3]] + [0.4])

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Plot contrasts from the constituent length experiments
    ''')
    args = argparser.parse_args()

    PLOT_ITEMS = True
    PLOT_LING = True
    PLOT_BAR = False
    PLOT_LINES = True
    DUMP_TEXT = False

    bar_width = 0.8
    capthick=1
    capsize=2
    x_sep = 1
    font_size = 16
    tick_size = 14
    legend_size = 10
    rotation = 30
    orangered = colors.to_rgba('orangered')
    dodgerblue = colors.to_rgba('dodgerblue')
    gray = colors.to_rgba('gray')
    gray = tuple([x if i < 3 else 0.4 for i, x in enumerate(gray)])
    plt.rcParams.update({'font.size': font_size, 'xtick.labelsize': tick_size, 'ytick.labelsize': tick_size})
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"

    x_width = 1
    n_network = 1

    colors_bycond = [
        # C
        (240, 202, 0, 255),
        (246, 158, 0, 255),
        (250, 122, 0, 255),
        (252, 90, 0, 255),
        (252, 66, 0, 255),
        (253,42,0,255),

        # NC
        (39, 196, 246, 255),
        (7, 124, 206, 255),

        # JAB
        (222, 153, 255, 255),
        (175, 133, 238, 255),
        (160, 82, 202, 255),
    ]
    colors_bycond = [tuple([float(x) / 255 for x in y]) for y in colors_bycond]

    colors_bylen = [
        # C
        (252, 66, 0, 255),

        # NC
        (7, 124, 206, 255),

        # JAB
        (160, 82, 202, 255),
    ]
    colors_bylen = [tuple([float(x) / 255 for x in y]) for y in colors_bylen]

    fROIs = [
        'LANGLIFGorb',
        'LANGLIFG',
        'LANGLMFG',
        'LANGLAntTemp',
        'LANGLPostTemp',
        'LANGLAngG'
    ]

    ling_names = {
        'wlen': 'Word Length',
        'unigramsurp': 'Unigram Surp',
        'fwprob5surp': '5-gram Surp',
        'totsurp': 'PCFG Surp',
        'PMI': 'PMI',
        'dltcvm': 'DLT (integration cost)',
        'dlts': 'DLT (storage cost)',
        'noF': 'End of constituent',
        'noFlen': 'Length of constituent',
        'noFlenlog1p': 'Log length of constituent',
        'opennodes': 'Open nodes',
        'nmerged': 'Nodes merged',
    }

    contrast_names = {
        'isCLen1': 'c01',
        'isCLen2': 'c02',
        'isCLen3': 'c03',
        'isCLen4': 'c04',
        'isCLen6': 'c06',
        'isCLen12': 'c12',
        'isNCLen3': 'nc03',
        'isNCLen4': 'nc04',
        'isJABLen1': 'jab-c01',
        'isJABLen4': 'jab-c04',
        'isJABLen12': 'jab-c12',
        'isC': 'c',
        'isNC': 'nc',
        'isJAB': 'jab',
        'CLen': 'c-len',
        'CLen34': 'c-len-3-4',
        'NCLen': 'nc-len',
        'JABLen': 'jab-len',
        'C>JAB': 'c > jab',
        'C>NC': 'c > nc',
        'CLen>JABLen': 'c-len > jab-len',
        'CLen>NCLen': 'c-len > nc-len'
    }

    cond2len = {
        'c01': 1,
        'c02': 2,
        'c03': 3,
        'c04': 4,
        'c06': 6,
        'c12': 12,
        'nc03': 3,
        'nc04': 4,
        'jab-c01': 1,
        'jab-c04': 4,
        'jab-c12': 12,
    }

    cond2type = {
        'c01': 'c',
        'c02': 'c',
        'c03': 'c',
        'c04': 'c',
        'c06': 'c',
        'c12': 'c',
        'nc03': 'nc',
        'nc04': 'nc',
        'jab-c01': 'jab',
        'jab-c04': 'jab',
        'jab-c12': 'jab',
    }

    cond2len_name = {
        'isCLen1': 'c01',
        'isCLen2': 'c02',
        'isCLen3': 'c03',
        'isCLen4': 'c04',
        'isCLen6': 'c06',
        'isCLen12': '12',
        'isNCLen3': 'c03',
        'isNCLen4': 'c04',
        'isJABLen1': 'c01',
        'isJABLen4': 'c04',
        'isJABLen12': 'c12',
    }

    cond2len_name2 = {
        'A_12c': 'c12',
        'B_6c': 'c06',
        'C_4c': 'c04',
        'E_3c': 'c03',
        'G_2c': 'c02',
        'H_1c': 'c01',
        'I_jab12c': 'jab-c12',
        'J_jab4c': 'jab-c04',
        'K_jab1c': 'jab-c01',
    }


    # Items

    itemmeasures = pd.read_csv('output/conlen/nlength_con2/conlen2.itemmeasures', sep=' ')
    preds_means = itemmeasures.groupby(['cond', 'conlen', 'docid']).mean().reset_index()
    preds_mean = itemmeasures.groupby(['cond', 'conlen']).mean()
    preds_err = itemmeasures.groupby(['cond', 'conlen']).sem()


    # PoS

    prefix = 'output/conlen/'
    pos = pd.read_csv('plot_data/pos.csv')
    means = {}
    lb = {}
    ub = {}
    for pos_tag, v in pos.groupby('Part of Speech'):
        v = v.sort_values('Length')
        means[pos_tag] = v['Mean'].values
        lb[pos_tag] = v['Mean'].values - v['2.5%'].values
        ub[pos_tag] =  v['97.5%'].values  - v['Mean'].values

    # SWJN1

    swjn_fROIs = {
        1: 'LANGLIFGorb',
        2: 'LANGLIFG',
        3: 'LANGLMFG',
        4: 'LANGLAntTemp',
        5: 'LANGLPostTemp',
        6: 'LANGLAngG'
    }

    df_swjn1_src = pd.read_csv('plot_data/SWJNV1_results.csv', sep=', *')
    df_swjn1 = []
    for fROI in swjn_fROIs:
        for contrast in ['S', 'W', 'J', 'N']:
            vals = df_swjn1_src[(df_swjn1_src.ROI == fROI) & (df_swjn1_src.Effect == contrast)].EffectSize.values
            df_swjn1.append((swjn_fROIs[fROI], contrast, vals.mean(), vals.std(axis=0) / np.sqrt(len(vals))))
    df_swjn1 = pd.DataFrame(df_swjn1, columns=['fROI', 'contrast', 'estimate', 'err'])

    # SWJN2

    df_swjn2_src = pd.read_csv('plot_data/SWJNV2_results.csv')
    df_swjn2 = []
    for fROI in swjn_fROIs:
        for contrast in ['S', 'W', 'J', 'N']:
            vals = df_swjn2_src[(df_swjn2_src.ROI == fROI) & (df_swjn2_src.Effect == contrast)].EffectSize.values
            df_swjn2.append((swjn_fROIs[fROI], contrast, vals.mean(), vals.std(axis=0) / np.sqrt(len(vals))))
    df_swjn2 = pd.DataFrame(df_swjn2, columns=['fROI', 'contrast', 'estimate', 'err'])

    # Localizer FED parcels

    loc_fROIs = {
        1: 'LANGLIFGorb',
        2: 'LANGLIFG',
        3: 'LANGLAntTemp',
        4: 'LANGLPostTemp',
        5: 'LANGLAngG'
    }
    df_fed_loc_src = pd.read_csv('parcel_comparison_data/spm_ss_mROI_data.conlen2_fed.csv', sep=', *')
    df_fed_loc_src.Effect = df_fed_loc_src.Effect.map(cond2len_name2)
    df_fed_loc = []
    for fROI in loc_fROIs:
        for contrast in ['c01', 'c02', 'c03', 'c04', 'c06', 'c12', 'jab-c01', 'jab-c04', 'jab-c12']:
            vals = df_fed_loc_src[(df_fed_loc_src.ROI == fROI) & (df_fed_loc_src.Effect == contrast)].EffectSize.values
            df_fed_loc.append((loc_fROIs[fROI], contrast, vals.mean(), vals.std(axis=0) / np.sqrt(len(vals))))
    df_fed_loc = pd.DataFrame(df_fed_loc, columns=['fROI', 'contrast', 'estimate', 'err'])

    # Localizer PDD parcels

    df_pdd_loc_src = pd.read_csv('parcel_comparison_data/spm_ss_mROI_data.conlen2_pdd.csv', sep=', *')
    df_pdd_loc_src.Effect = df_pdd_loc_src.Effect.map(cond2len_name2)
    df_pdd_loc = []
    for fROI in loc_fROIs:
        for contrast in ['c01', 'c02', 'c03', 'c04', 'c06', 'c12', 'jab-c01', 'jab-c04', 'jab-c12']:
            vals = df_pdd_loc_src[(df_pdd_loc_src.ROI == fROI) & (df_pdd_loc_src.Effect == contrast)].EffectSize.values
            df_pdd_loc.append((loc_fROIs[fROI], contrast, vals.mean(), vals.std(axis=0) / np.sqrt(len(vals))))
    df_pdd_loc = pd.DataFrame(df_pdd_loc, columns=['fROI', 'contrast', 'estimate', 'err'])

    # Toolbox PDD estimates

    df_pdd_src = pd.read_csv('plot_data/Nlength_con2_results.csv')
    pdd_fROIs = {
        1: 'LANGLIFGorb',
        2: 'LANGLIFG',
        3: 'LANGLMFG',
        4: 'LANGLAntTemp',
        5: 'LANGLPostTemp',
        6: 'LANGLAngG'
    }

    def pdd_effect_mapper(x):
        x = x[2:]
        if x == '12c':
            return 'S'
        if x == '6c':
            return 'c06'
        if x == '4c':
            return 'c04'
        if x == '3c':
            return 'c03'
        if x == '2c':
            return 'c02'
        if x == '1c':
            return 'W'
        if x == '4nc':
            return 'nc04'
        if x == '3nc':
            return 'nc03'
        if x == 'jab12c':
            return 'J'
        if x == 'jab4c':
            return 'jab-c04'
        if x == 'jab1c':
            return 'N'
        raise ValueError('Unrecognized value %s in PDD mapper fn' % x)


    df_pdd_src['Condition'] = df_pdd_src.Effect.apply(pdd_effect_mapper)
    df_pdd_src = df_pdd_src.groupby(['Subject', 'ROI', 'Condition']).mean().reset_index()
    df_pdd_src = df_pdd_src[df_pdd_src.ROI < 7]

    df_pdd = []
    for fROI in pdd_fROIs:
        for contrast in ['S', 'W', 'J', 'N']:
            vals = df_pdd_src[(df_pdd_src.ROI == fROI) & (df_pdd_src.Condition == contrast)].EffectSize.values
            df_pdd.append((pdd_fROIs[fROI], contrast, vals.mean(), vals.std(axis=0) / np.sqrt(len(vals))))
    df_pdd = pd.DataFrame(df_pdd, columns=['fROI', 'contrast', 'estimate', 'err'])



    # Plotting


    # PDD vs SWJN

    fig = plt.figure(figsize=((13, 6.2)))
    h = [Size.Fixed(0.7), Size.Fixed(10.3)]
    v = [Size.Fixed(1.5), Size.Fixed(4.1)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=1, ny=1)
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.tick_params(labelleft='on', labelbottom='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3, zorder=1)
    ax.axhline(y=0, lw=1, c='gray', alpha=1, zorder=2)

    color = [
        # S
        (137, 69, 246, 255),

        # W
        (14, 60, 245, 255),

        # J
        (235, 63, 37, 255),

        # N
        (192, 192, 192, 255)
    ]
    color = [tuple([float(x) / 255 for x in y]) for y in color]

    bar_width = 1./12 * 0.7
    contrasts = ['S', 'W', 'J', 'N']

    for i in range(1, 6):
        ax.axvline(x=i - (1.5 * bar_width), lw=1, c='gray', alpha=1, zorder=2)

    r_base = np.arange(6)
    for i in range(12):
        j = i % 4
        # if i % 2 == 0:
        if i < 4:
            _df = df_swjn1
        elif i >= 8:
            _df = df_pdd
        else:
            _df = df_swjn2
        contrast = contrasts[j]
        df = []
        df.append(_df[(_df.contrast == contrast) & (_df.fROI != 'all')])
        df = pd.concat(df, axis=0)
        df = df.to_dict('records')
        df = sorted(df, key=lambda x: {'LANGLIFGorb': 1, 'LANGLIFG': 2, 'LANGLMFG': 3, 'LANGLAntTemp': 4, 'LANGLPostTemp': 5, 'LANGLAngG': 6}[x['fROI']])
        df = pd.DataFrame(df)
        r = r_base + i * bar_width + (i // 4) * 0.1

        estimates = df.estimate.values
        errors = df.err.values

        # # 3.42 is ~ratio of HRF integrals in pyMVPA vs SPM
        # estimates /= 3.42
        # errors /= 3.42

        ax.bar(
            r,
            estimates,
            color=color[j],
            width=bar_width,
            label=contrast if i < 4 else None,
            linewidth=2,
            # linestyle='solid' if (i % 2 == 0) else 'dashed'
        )

        ax.errorbar(
            r,
            estimates,
            yerr=errors,
            fmt='none',
            ecolor=color[j],
            capsize=2,
            capthick=2,
            linewidth=2
        )

    ax.legend(loc='center left', ncol=1, bbox_to_anchor=(1, 0.5))

    ax.set_xticks(np.arange(0, 5.9, 0.3333333) + bar_width * 1.5)
    ax.set_xticklabels(['F. et al (2010) Exp1', 'F. et al (2010) Exp2', 'Current Study'] * 6, rotation=45, ha='right')
    # ax.set_ylabel('BOLD')
    ax.set_ylim(-1.37, 6)

    plt.savefig('output/conlen/plots/PDD_SWJN.png')



    # FED vs PDD parcels

    fig = plt.figure(figsize=((13, 6.2)))
    h = [Size.Fixed(0.7), Size.Fixed(10.3)]
    v = [Size.Fixed(1.5), Size.Fixed(4.1)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=1, ny=1)
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.tick_params(labelleft='on', labelbottom='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3, zorder=1)
    ax.axhline(y=0, lw=1, c='gray', alpha=1, zorder=2)

    color = [
        # C
        (240, 202, 0, 255),
        (246, 158, 0, 255),
        (250, 122, 0, 255),
        (252, 90, 0, 255),
        (252, 66, 0, 255),
        (253,42,0,255),

        # JAB
        (222, 153, 255, 255),
        (175, 133, 238, 255),
        (160, 82, 202, 255),
    ]
    color = [tuple([float(x) / 255 for x in y]) for y in color]

    bar_width = 1./18 * 0.8
    contrasts = ['c01', 'c02', 'c03', 'c04', 'c06', 'c12', 'jab-c01', 'jab-c04', 'jab-c12']

    for i in range(1, 5):
        ax.axvline(x=i - 1.5 * bar_width, lw=1, c='gray', alpha=1, zorder=2)

    r_base = np.arange(5)
    for i in range(18):
        j = i % 9
        # if i % 2 == 0:
        if i < 9:
            _df = df_pdd_loc
        else:
            _df = df_fed_loc
        contrast = contrasts[j]
        df = []
        df.append(_df[(_df.contrast == contrast) & (_df.fROI != 'all')])
        df = pd.concat(df, axis=0)
        df = df.to_dict('records')
        df = sorted(df, key=lambda x: {'LANGLIFGorb': 1, 'LANGLIFG': 2, 'LANGLAntTemp': 3, 'LANGLPostTemp': 4, 'LANGLAngG': 5}[x['fROI']])
        df = pd.DataFrame(df)
        r = r_base + i * bar_width + (i // 9) * 0.1

        estimates = df.estimate.values
        errors = df.err.values

        # # 3.42 is ~ratio of HRF integrals in pyMVPA vs SPM
        # estimates /= 3.42
        # errors /= 3.42

        ax.bar(
            r,
            estimates,
            color=color[j],
            width=bar_width,
            label=contrast if i < 9 else None,
            linewidth=2,
            # linestyle='solid' if (i % 2 == 0) else 'dashed'
        )

        ax.errorbar(
            r,
            estimates,
            yerr=errors,
            fmt='none',
            ecolor=color[j],
            capsize=2,
            capthick=2,
            linewidth=2
        )

    ax.legend(loc='center left', ncol=1, bbox_to_anchor=(1, 0.5))

    ax.set_xticks(np.arange(0, 5, 0.5) + bar_width * 4)
    ax.set_xticklabels(['PDD parcels', 'Our parcels'] * 5, rotation=45, ha='right')
    # ax.set_ylabel('BOLD')
    ax.set_ylim(-1.37, 6)

    plt.savefig('output/conlen/plots/parcel_comparison.png')




    # POS


    fig = plt.figure(figsize=(10,3.1))
    h = [Size.Fixed(1.0), Size.Fixed(6)]
    v = [Size.Fixed(0.6), Size.Fixed(2.5)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=1, ny=1)
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(labelleft='on', labelbottom='on')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('bottom')
    # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
    # ax.axhline(y=0, lw=1, c='gray', alpha=1)

    hatches = [
        '///',
        '...',
        '++',
        'oo'
    ]

    left = np.zeros(6)
    for i, pos_tag in enumerate(['Adjective/Adverb', 'Verb', 'Noun', 'Function Word']):
        ax.barh(
            np.arange(5,-1,-1),
            means[pos_tag],
            left=left,
            # color=[get_color(x, c) for x in names],
            color='white',
            edgecolor='gray',
            hatch=hatches[i],
            lw=1.5,
            label=pos_tag
        )
        # ax.errorbar(
        #     means[pos_tag] + left,
        #     np.arange(5,-1,-1),
        #     xerr=np.stack([lb[pos_tag], ub[pos_tag]], axis=0),
        #     fmt='none',
        #     ecolor='black',
        #     lw=2,
        #     capthick=capthick,
        #     capsize=capsize
        # )
        left += means[pos_tag]

    ax.set_yticks(np.arange(5,-1,-1))
    ax.set_yticklabels(['c01', 'c02', 'c03', 'c04', 'c06', 'c12'])
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    # ax.set_xlabel('Length Condition')
    ax.set_xlabel('Proportion of words')

    if not os.path.exists(prefix + 'plots'):
        os.makedirs(prefix + 'plots')

    plt.savefig(prefix + 'plots/' + 'pos_distribution.png')




    # Experiment 1

    plot_data = {}

    experiment = '1'

    plot_data[experiment] = {}

    prefix = 'output/conlen/nlength_con%s/' % experiment

    baseline_means = {}
    baseline_errs = {}

    colors_bycond_cur = [
        colors_bycond[0],
        colors_bycond[1],
        colors_bycond[3],
        colors_bycond[4],
        colors_bycond[5],
    ]

    for path in [prefix + 'contrasts/' + x for x in os.listdir(prefix + 'contrasts/') if x.endswith('.csv')]:
        baseline_key = path.split('.')[-2]
        baseline_key_str = baseline_key

        plot_data[experiment][baseline_key_str] = {}

        # Data
        df = pd.read_csv(path, index_col=0)

        by_froi = {}

        names = None

        for k, v in df.groupby(['subject', 'fROI', 'ling']):
            subject, fROI, ling = k
            if ling not in by_froi:
                by_froi[ling] = {}
            if fROI not in by_froi[ling]:
                by_froi[ling][fROI] = []
            contrasts_cur = v.copy()
            contrasts_cur = contrasts_cur[~contrasts_cur.contrast.isin(['isC34', 'isC1412', 'CLen34', 'CLen1412'])]
            if names is None:
                names = contrasts_cur.contrast.to_list()

            by_froi[ling][fROI].append(contrasts_cur.estimate.values)

        means = {}
        errs = {}

        names = [contrast_names[x] if x in contrast_names else x for x in names]
        names_ax = names[:]
        names_ax.insert(2, 'C3')

        contrasts_all = {}
        for ling in by_froi:
            _contrasts_all = []
            for fROI in by_froi[ling]:
                contrasts = np.stack(by_froi[ling][fROI], axis=0)
                contrast_mean = contrasts.mean(axis=0)
                contrast_sem = contrasts.std(axis=0) / np.sqrt(len(contrasts))
                if ling not in means:
                    means[ling] = {}
                if ling not in errs:
                    errs[ling] = {}
                means[ling][fROI] = contrast_mean
                errs[ling][fROI] = contrast_sem

                _contrasts_all.append(contrasts)

            _contrasts_all = np.concatenate(_contrasts_all, axis=0)
            _contrasts_all_means = _contrasts_all.mean(axis=0)
            _contrasts_all_sem = _contrasts_all.std(axis=0) / np.sqrt(len(_contrasts_all))
            contrasts_all[ling] = {
                'all': pd.DataFrame({
                    'contrast': names,
                    'estimate': _contrasts_all_means,
                    'err': _contrasts_all_sem,
                })
            }

        plot_data[experiment][baseline_key_str] = contrasts_all

        # By condition

        r_base = np.concatenate([
            np.arange(0, 2),
            np.arange(3, 6),
            np.arange(0, 1),
            np.arange(6, 7) + x_sep
        ], axis=0)

        r_base_ticks = np.concatenate([
            np.arange(6),
            np.arange(0, 1),
            np.arange(6, 7) + x_sep
        ], axis=0)

        r = r_base * x_width
        r_ticks = r_base_ticks * x_width
        clip = 6

        for ling in means:
        # for ling in ['none']:
            fig = plt.figure()
            axes = []
            for i in range(6):
                axes.append((fROIs[i], fig.add_subplot(6, 1, i + 1)))

            for i, (fROI, ax) in enumerate(axes):
                mean = means[ling][fROI]
                err = errs[ling][fROI]

                plot_data[experiment][baseline_key_str][ling][fROI] = pd.DataFrame({
                    'contrast': names,
                    'estimate': mean,
                    'err': err,
                })

                _r = r[:5]
                _r_ticks = r_ticks[:6]
                _mean = mean[:5]
                _err = err[:5]
                _names = names[:5]
                _names_ax = names_ax[:6]
                if ling == 'none' and baseline_key_str != 'none':
                    if PLOT_LING:
                        _r = np.concatenate([_r, [r[-1]]], axis=0)
                        _r_ticks = np.concatenate([_r_ticks, [r_ticks[-1]]], axis=0)
                        _mean = np.concatenate([_mean, [mean[-1]]], axis=0)
                        _err = np.concatenate([_err, [err[-1]]], axis=0)
                        _names.append(ling_names[baseline_key_str])
                        _names_ax.append(ling_names[baseline_key_str])
                        color = colors_bycond
                        clip = 7
                    else:
                        color = colors_bycond_cur
                    if fROI not in baseline_means:
                        baseline_means[fROI] = {}
                    baseline_means[fROI][baseline_key_str] = mean[-1]
                    if fROI not in baseline_errs:
                        baseline_errs[fROI] = {}
                    baseline_errs[fROI][baseline_key_str] = err[-1]
                else:
                    color = colors_bycond_cur

                _df = pd.DataFrame({
                    'contrast': _names,
                    'estimate': _mean,
                    'err': _err,
                })

                if DUMP_TEXT:
                    _df.to_csv(prefix + 'plots/' + 'bycond.%s.baseline_%s.%s.txt' % (ling, baseline_key_str, fROI), index=False, sep='\t')

                if PLOT_BAR:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.tick_params(labelleft='on', labelbottom='on')
                    ax.yaxis.set_ticks_position('none')
                    ax.xaxis.set_ticks_position('bottom')
                    # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
                    ax.axhline(y=0, lw=1, c='gray', alpha=1)

                    ax.bar(
                        _r,
                        _mean,
                        # color=[get_color(x, c) for x in names],
                        color=color,
                        edgecolor='none',
                        lw=1.5,
                        label=_names,
                    )

                    for j, x in enumerate(_names):
                        ax.errorbar(
                            _r[j:j + 1],
                            _mean[j:j + 1],
                            yerr=_err[j:j + 1],
                            fmt='none',
                            ecolor='black',
                            lw=2,
                            capthick=capthick,
                            capsize=capsize
                        )

                    # plt.xlabel('Effect')
                    ax.set_xticks(_r_ticks)
                    if i == 5:
                        ax.set_xticklabels(_names_ax, rotation=rotation, ha='right')
                    else:
                        ax.set_xticklabels([])
                    if fROI == 'Mean':
                        ax.set_title(fROI)
                    else:
                        ax.set_title(fROI[4:])

            if PLOT_BAR:
                # if baseline_key != 'none':
                #     fig.suptitle('Exp %s, %s Controlled' % (experiment, ling_names.get(baseline_key, baseline_key)))
                fig.set_size_inches(clip, 12)
                fig.tight_layout()

                if not os.path.exists(prefix + 'plots'):
                    os.makedirs(prefix + 'plots')

                plt.savefig(prefix + 'plots/' + 'bycond.%s.baseline_%s.png' % (ling, baseline_key_str))
            plt.close('all')

        # Overall length effects

        if PLOT_BAR or DUMP_TEXT:
            for ling in means:
            # for ling in ['none']:
                fig = plt.figure()
                axes = []
                for i in range(6):
                    axes.append((fROIs[i], fig.add_subplot(6, 1, i + 1)))

                for i, (fROI, ax) in enumerate(axes):
                    mean = means[ling][fROI]
                    err = errs[ling][fROI]

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.tick_params(labelleft='on', labelbottom='on')
                    ax.yaxis.set_ticks_position('none')
                    ax.xaxis.set_ticks_position('bottom')
                    # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
                    ax.axhline(y=0, lw=1, c='gray', alpha=1)

                    _df = pd.DataFrame({
                        'contrast': names[5:6],
                        'estimate': mean[5:6],
                        'err': err[5:6]
                    })

                    if DUMP_TEXT:
                        _df.to_csv(prefix + 'plots/' + 'bylen.%s.baseline_%s.%s.txt' % (ling, baseline_key_str, fROI), index=False, sep='\t')

                    if PLOT_BAR:
                        ax.bar(
                            r[5:6],
                            mean[5:6],
                            # color=[get_color(x, c) for x in names],
                            color=[colors_bylen[0]],
                            edgecolor='none',
                            lw=1.5,
                            label=names[5:6],
                        )

                        for j, x in enumerate(names[5:6]):
                            ax.errorbar(
                                r[j + 5:j + 6],
                                mean[j + 5:j + 6],
                                yerr=err[j + 5:j + 6],
                                fmt='none',
                                ecolor='black',
                                lw=2,
                                capthick=capthick,
                                capsize=capsize
                            )

                        # plt.xlabel('Effect')
                        ax.set_xticks(r_base[6:7] * x_width)
                        if i == 5:
                            ax.set_xticklabels(names_ax[6:7], rotation=rotation, ha='right')
                        else:
                            ax.set_xticklabels([])
                        if fROI == 'Mean':
                            ax.set_title(fROI)
                        else:
                            ax.set_title(fROI[4:])


                if PLOT_BAR:
                    # if baseline_key != 'none':
                    #     fig.suptitle('Exp %s, %s Controlled' % (experiment, ling_names.get(baseline_key, baseline_key)))
                    fig.set_size_inches(3, 12)
                    fig.tight_layout()

                    plt.savefig(prefix + 'plots/' + 'bylen.%s.baseline_%s.png' % (ling, baseline_key_str))
                plt.close('all')


    # Baseline effects
    fig = plt.figure()
    axes = []
    for i in range(6):
        axes.append((fROIs[i], fig.add_subplot(6, 1, i + 1)))


    if PLOT_BAR or DUMP_TEXT:
        for i, (fROI, ax) in enumerate(axes):
            pred_keys = sorted(baseline_means[fROI].keys())
            _mean = [baseline_means[fROI][x] for x in pred_keys]
            _err = [baseline_errs[fROI][x] for x in pred_keys]
            _names = [ling_names[x] for x in pred_keys]
            _r = np.arange(len(pred_keys)) * x_width

            _df = pd.DataFrame({
                'contrast': _names,
                'estimate': _mean,
                'err': _err
            })
            if DUMP_TEXT:
                _df.to_csv(prefix + 'plots/' + 'baseline_estimates.%s.txt' % fROI, index=False, sep='\t')

            if PLOT_BAR:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(labelleft='on', labelbottom='on')
                ax.yaxis.set_ticks_position('none')
                ax.xaxis.set_ticks_position('bottom')
                # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
                ax.axhline(y=0, lw=1, c='gray', alpha=1)

                cmap = plt.get_cmap('gist_rainbow')

                ax.bar(
                    _r,
                    _mean,
                    color=[cmap(float(j) / clip) for j in range(clip)],
                    edgecolor='none',
                    lw=1.5,
                    label=_names,
                )

                for j, x in enumerate(_names):
                    ax.errorbar(
                        _r[j:j + 1],
                        _mean[j:j + 1],
                        yerr=_err[j:j + 1],
                        fmt='none',
                        ecolor='black',
                        lw=2,
                        capthick=capthick,
                        capsize=capsize
                    )

                # plt.xlabel('Effect')
                ax.set_xticks(_r)
                if i == 5:
                    ax.set_xticklabels(_names, rotation=rotation, ha='right')
                else:
                    ax.set_xticklabels([])
                if fROI == 'Mean':
                    ax.set_title(fROI)
                else:
                    ax.set_title(fROI[4:])

        if PLOT_BAR:
            fig.set_size_inches(6, 12)
            fig.tight_layout()

            plt.savefig(prefix + 'plots/' + 'baseline_estimates.png')
        plt.close('all')


    # Experiment 2

    experiment = '2'

    plot_data[experiment] = {}

    fROIs = [
        'LANGLIFGorb',
        'LANGLIFG',
        'LANGLMFG',
        'LANGLAntTemp',
        'LANGLPostTemp',
        'LANGLAngG'
    ]

    plt.rcParams.update({'font.size': font_size, 'xtick.labelsize': tick_size})

    prefix = 'output/conlen/nlength_con%s/' % experiment

    baseline_means = {}
    baseline_errs = {}

    if PLOT_ITEMS:
        fig = plt.figure(figsize=((3, 9./4)))
        h = [Size.Fixed(0.5), Size.Fixed(2.5)]
        v = [Size.Fixed(0.5), Size.Fixed(7./4)]
        divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
        ax = fig.add_axes(
            divider.get_position(),
            axes_locator=divider.new_locator(nx=1, ny=1)
        )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(labelleft='on', labelbottom='on')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
        # ax.axhline(y=0, lw=1, c='gray', alpha=1)

        ax.barh(
            np.arange(5,-1,-1),
            np.arange(6),
            # color=[get_color(x, c) for x in names],
            color=colors_bycond,
            edgecolor='none',
            lw=1.5,
            label=['c12', 'c06', 'c04', 'c03', 'c02', 'c01'],
        )

        # plt.xlabel('Effect')
        ax.set_yticks(np.arange(5,-1,-1) * x_width)
        ax.set_yticklabels(['c01', 'c02', 'c03', 'c04', 'c06', 'c12'], rotation=rotation, ha='right')
        # ax.set_ylabel(ling_names.get(baseline_key, baseline_key))

        if not os.path.exists(prefix + 'plots'):
            os.makedirs(prefix + 'plots')

        plt.savefig('output/conlen/plots/items.pdd.png')
        plt.close('all')

    for path in [prefix + 'contrasts/' + x for x in os.listdir(prefix + 'contrasts/') if x.endswith('.csv')]:
        baseline_key = path.split('.')[-2]
        baseline_key_str = baseline_key

        plot_data[experiment][baseline_key_str] = {}

        if baseline_key != 'none':
            clip = 6
        else:
            clip = 11

        # Data
        df = pd.read_csv(path, index_col=0)

        by_froi = {}

        names = None

        for k, v in df.groupby(['subject', 'fROI', 'ling']):
            subject, fROI, ling = k
            if ling not in by_froi:
                by_froi[ling] = {}
            if fROI not in by_froi[ling]:
                by_froi[ling][fROI] = []
            contrasts_cur = v.copy()
            if names is None:
                names = contrasts_cur.contrast.to_list()

            by_froi[ling][fROI].append(contrasts_cur.estimate.values)

        means = {}
        errs = {}

        names = [contrast_names[x] if x in contrast_names else x for x in names]

        contrasts_all = {}
        for ling in by_froi:
            _contrasts_all = []
            for fROI in by_froi[ling]:
                contrasts = np.stack(by_froi[ling][fROI], axis=0)
                contrast_mean = contrasts.mean(axis=0)
                contrast_sem = contrasts.std(axis=0) / np.sqrt(len(contrasts))
                if ling not in means:
                    means[ling] = {}
                if ling not in errs:
                    errs[ling] = {}
                means[ling][fROI] = contrast_mean
                errs[ling][fROI] = contrast_sem

                _contrasts_all.append(contrasts)

            _contrasts_all = np.concatenate(_contrasts_all, axis=0)
            _contrasts_all_means = _contrasts_all.mean(axis=0)
            _contrasts_all_sem = _contrasts_all.std(axis=0) / np.sqrt(len(_contrasts_all))
            contrasts_all[ling] = {
                'all': pd.DataFrame({
                    'contrast': names,
                    'estimate': _contrasts_all_means,
                    'err': _contrasts_all_sem,
                })
            }

        plot_data[experiment][baseline_key_str] = contrasts_all


        # By condition

        r_base = np.concatenate([
            np.arange(6),
            np.arange(6, 8) + x_sep,
            np.arange(8, 11) + x_sep * 2,
            np.arange(0, 3),
            np.arange(0, 3),
            np.arange(0, 2),
            np.arange(0, 2),
            np.arange(6, 7) + x_sep,
        ], axis=0)

        r = r_base * x_width

        if baseline_key != 'none' and (PLOT_ITEMS or DUMP_TEXT):
            fig = plt.figure(figsize=((3, 9./4)))
            h = [Size.Fixed(0.5), Size.Fixed(2.5)]
            v = [Size.Fixed(0.5), Size.Fixed(7./4)]
            divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
            ax = fig.add_axes(
                divider.get_position(),
                axes_locator=divider.new_locator(nx=1, ny=1)
            )

            mean = preds_mean.loc[:, baseline_key]
            err = preds_err.loc[:, baseline_key]

            _df = pd.DataFrame({
                'contrast': names[:clip],
                'estimate': mean[:clip],
                'err': err[:clip],
            })

            if DUMP_TEXT:
                _df.to_csv(prefix + 'plots/' + 'items.%s.txt' % baseline_key, index=False, sep='\t')

            if PLOT_ITEMS:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(labelleft='on', labelbottom='on')
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
                # ax.axhline(y=0, lw=1, c='gray', alpha=1)

                ax.barh(
                    r[:clip][::-1],
                    mean[:clip],
                    # color=[get_color(x, c) for x in names],
                    color=colors_bycond,
                    edgecolor='none',
                    lw=1.5,
                    label=names[:clip],
                )

                for j, x in enumerate(names[:clip]):
                    ax.errorbar(
                        mean[j:j + 1],
                        r[5-j:5-j + 1],
                        xerr=err[j:j + 1],
                        fmt='none',
                        ecolor='black',
                        lw=2,
                        capthick=capthick,
                        capsize=capsize
                    )

                # plt.xlabel('Effect')
                ax.set_yticks(r_base[:clip][::-1] * x_width)
                ax.set_yticklabels(names[:clip], rotation=rotation, ha='right')
                # ax.set_ylabel(ling_names.get(baseline_key, baseline_key))

                if not os.path.exists(prefix + 'plots'):
                    os.makedirs(prefix + 'plots')

                plt.savefig('output/conlen/plots/items.%s.png' % baseline_key)
            plt.close('all')

        for ling in means:
            fig = plt.figure()
            axes = []
            for i in range(6):
                axes.append((fROIs[i], fig.add_subplot(6, 1, i + 1)))

            for i, (fROI, ax) in enumerate(axes):
                mean = means[ling][fROI]
                err = errs[ling][fROI]

                plot_data[experiment][baseline_key_str][ling][fROI] = pd.DataFrame({
                    'contrast': names,
                    'estimate': mean,
                    'err': err,
                })

                _r = r[:clip]
                _mean = mean[:clip]
                _err = err[:clip]
                _names = names[:clip]
                if ling == 'none' and baseline_key_str != 'none':
                    if PLOT_LING:
                        _r = np.concatenate([_r, [r[-1]]], axis=0)
                        _mean = np.concatenate([_mean, [mean[-1]]], axis=0)
                        _err = np.concatenate([_err, [err[-1]]], axis=0)
                        _names.append(ling_names[baseline_key_str])
                    if fROI not in baseline_means:
                        baseline_means[fROI] = {}
                    baseline_means[fROI][baseline_key_str] = mean[-1]
                    if fROI not in baseline_errs:
                        baseline_errs[fROI] = {}
                    baseline_errs[fROI][baseline_key_str] = err[-1]

                _df = pd.DataFrame({
                    'contrast': _names,
                    'estimate': _mean,
                    'err': _err
                })

                if DUMP_TEXT:
                    _df.to_csv(prefix + 'plots/' + 'bycond.%s.baseline_%s.%s.txt' % (ling, baseline_key_str, fROI), index=False, sep='\t')

                if PLOT_BAR:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.tick_params(labelleft='on', labelbottom='on')
                    ax.yaxis.set_ticks_position('none')
                    ax.xaxis.set_ticks_position('bottom')
                    # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
                    ax.axhline(y=0, lw=1, c='gray', alpha=1)

                    ax.bar(
                        _r,
                        _mean,
                        # color=[get_color(x, c) for x in names],
                        color=colors_bycond,
                        edgecolor='none',
                        lw=1.5,
                        label=_names,
                    )

                    for j, x in enumerate(names[:clip]):
                        ax.errorbar(
                            _r,
                            _mean,
                            yerr=_err,
                            fmt='none',
                            ecolor='black',
                            lw=2,
                            capthick=capthick,
                            capsize=capsize
                        )

                    # plt.xlabel('Effect')
                    ax.set_xticks(_r)
                    if i == 5:
                        ax.set_xticklabels(_names, rotation=rotation, ha='right')
                    else:
                        ax.set_xticklabels([])
                    if fROI == 'Mean':
                        ax.set_title(fROI)
                    else:
                        ax.set_title(fROI[4:])

            if PLOT_BAR:
                # if baseline_key != 'none':
                #     fig.suptitle('Exp %s, %s Controlled' % (experiment, ling_names.get(baseline_key, baseline_key)))
                fig.set_size_inches(clip, 12)
                fig.tight_layout()

                if not os.path.exists(prefix + 'plots'):
                    os.makedirs(prefix + 'plots')

                plt.savefig(prefix + 'plots/' + 'bycond.%s.baseline_%s.png' % (ling, baseline_key_str))
            plt.close('all')



        # Overall stimulus type effects

        if PLOT_BAR or DUMP_TEXT:
            for ling in means:
                fig = plt.figure()
                axes = []
                for i in range(6):
                    axes.append((fROIs[i], fig.add_subplot(6, 1, i + 1)))

                for i, (fROI, ax) in enumerate(axes):
                    mean = means[ling][fROI]
                    err = errs[ling][fROI]

                    _df = pd.DataFrame({
                        'contrast': names[11:14],
                        'estimate': mean[11:14],
                        'err': err[11:14]
                    })

                    if DUMP_TEXT:
                        _df.to_csv(prefix + 'plots/' + 'bystimtype.%s.baseline_%s.%s.txt' % (ling, baseline_key_str, fROI), index=False, sep='\t')

                    if PLOT_BAR:
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.tick_params(labelleft='on', labelbottom='on')
                        ax.yaxis.set_ticks_position('none')
                        ax.xaxis.set_ticks_position('bottom')
                        # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
                        ax.axhline(y=0, lw=1, c='gray', alpha=1)

                        ax.bar(
                            r[11:14],
                            mean[11:14],
                            # color=[get_color(x, c) for x in names],
                            color=colors_bylen,
                            edgecolor='none',
                            lw=1.5,
                            label=names[11:14],
                        )

                        for j, x in enumerate(names[11:14]):
                            ax.errorbar(
                                r[j+11:j+12],
                                mean[j+11:j+12],
                                yerr=err[j+11:j+12],
                                fmt='none',
                                ecolor='black',
                                lw=2,
                                capthick=capthick,
                                capsize=capsize
                            )

                        # plt.xlabel('Effect')
                        ax.set_xticks(r_base[11:14] * x_width)
                        if i == 5:
                            ax.set_xticklabels(names[11:14], rotation=rotation, ha='right')
                        else:
                            ax.set_xticklabels([])
                        if fROI == 'Mean':
                            ax.set_title(fROI)
                        else:
                            ax.set_title(fROI[4:])

                if PLOT_BAR:
                    # if baseline_key != 'none':
                    #     fig.suptitle('Exp %s, %s Controlled' % (experiment, ling_names.get(baseline_key, baseline_key)))
                    fig.set_size_inches(3, 12)
                    fig.tight_layout()

                    plt.savefig(prefix + 'plots/' + 'bystimtype.%s.baseline_%s.png' % (ling, baseline_key_str))
                plt.close('all')




        # Overall length effects

        if PLOT_BAR or DUMP_TEXT:
            for ling in means:
                fig = plt.figure()
                axes = []
                for i in range(6):
                    axes.append((fROIs[i], fig.add_subplot(6, 1, i + 1)))

                for i, (fROI, ax) in enumerate(axes):
                    mean = means[ling][fROI]
                    err = errs[ling][fROI]

                    _df = pd.DataFrame({
                        'contrast': names[11:14],
                        'estimate': mean[11:14],
                        'err': err[11:14]
                    })

                    if DUMP_TEXT:
                        _df.to_csv(prefix + 'plots/' + 'bylen.%s.baseline_%s.%s.txt' % (ling, baseline_key_str, fROI), index=False, sep='\t')

                    if PLOT_BAR:
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.tick_params(labelleft='on', labelbottom='on')
                        ax.yaxis.set_ticks_position('none')
                        ax.xaxis.set_ticks_position('bottom')
                        # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
                        ax.axhline(y=0, lw=1, c='gray', alpha=1)

                        ax.bar(
                            r[14:17],
                            mean[14:17],
                            # color=[get_color(x, c) for x in names],
                            color=colors_bylen,
                            edgecolor='none',
                            lw=1.5,
                            label=names[14:17],
                        )

                        for j, x in enumerate(names[14:17]):
                            ax.errorbar(
                                r[j+14:j+15],
                                mean[j+14:j+15],
                                yerr=err[j+14:j+15],
                                fmt='none',
                                ecolor='black',
                                lw=2,
                                capthick=capthick,
                                capsize=capsize
                            )

                        # plt.xlabel('Effect')
                        ax.set_xticks(r_base[14:17] * x_width)
                        if i == 5:
                            ax.set_xticklabels(names[14:17], rotation=rotation, ha='right')
                        else:
                            ax.set_xticklabels([])
                        if fROI == 'Mean':
                            ax.set_title(fROI)
                        else:
                            ax.set_title(fROI[4:])

                if PLOT_BAR:
                    # if baseline_key != 'none':
                    #     fig.suptitle('Exp %s, %s Controlled' % (experiment, ling_names.get(baseline_key, baseline_key)))
                    fig.set_size_inches(3, 12)
                    fig.tight_layout()

                    plt.savefig(prefix + 'plots/' + 'bylen.%s.baseline_%s.png' % (ling, baseline_key_str))
                plt.close('all')



        # Interaction effects (stimulus type)

        if PLOT_BAR or DUMP_TEXT:
            for ling in means:
                fig = plt.figure()
                axes = []
                for i in range(6):
                    axes.append((fROIs[i], fig.add_subplot(6, 1, i + 1)))

                for i, (fROI, ax) in enumerate(axes):
                    mean = means[ling][fROI]
                    err = errs[ling][fROI]

                    _df = pd.DataFrame({
                        'contrast': names[17:19],
                        'estimate': mean[17:19],
                        'err': err[17:19]
                    })

                    if DUMP_TEXT:
                        _df.to_csv(prefix + 'plots/' + 'interaction.stimtype.%s.baseline_%s.%s.txt' % (ling, baseline_key_str, fROI), index=False, sep='\t')

                    if PLOT_BAR:
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.tick_params(labelleft='on', labelbottom='on')
                        ax.yaxis.set_ticks_position('none')
                        ax.xaxis.set_ticks_position('bottom')
                        # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
                        ax.axhline(y=0, lw=1, c='gray', alpha=1)

                        colors_bylen_cur = colors_bylen[1:]

                        ax.bar(
                            r[17:19],
                            mean[17:19],
                            # color=[get_color(x, c) for x in names],
                            color=colors_bylen_cur,
                            edgecolor='none',
                            lw=1.5,
                            label=names[17:19],
                        )

                        for j, x in enumerate(names[17:19]):
                            ax.errorbar(
                                r[j+17:j+18],
                                mean[j+17:j+18],
                                yerr=err[j+17:j+18],
                                fmt='none',
                                ecolor='black',
                                lw=2,
                                capthick=capthick,
                                capsize=capsize
                            )

                        # plt.xlabel('Effect')
                        ax.set_xticks(r_base[17:19] * x_width)
                        if i == 5:
                            ax.set_xticklabels(names[17:19], rotation=rotation, ha='right')
                        else:
                            ax.set_xticklabels([])
                        if fROI == 'Mean':
                            ax.set_title(fROI)
                        else:
                            ax.set_title(fROI[4:])

                if PLOT_BAR:
                    # if baseline_key != 'none':
                    #     fig.suptitle('Exp %s, %s Controlled' % (experiment, ling_names.get(baseline_key, baseline_key)))
                    fig.set_size_inches(3, 12)
                    fig.tight_layout()

                plt.savefig(prefix + 'plots/' + 'interaction.stimtype.%s.baseline_%s.png' % (ling, baseline_key_str))
                plt.close('all')



        # Interaction effects (length)

        if PLOT_BAR or DUMP_TEXT:
            for ling in means:
                fig = plt.figure()
                axes = []
                for i in range(6):
                    axes.append((fROIs[i], fig.add_subplot(6, 1, i + 1)))

                for i, (fROI, ax) in enumerate(axes):
                    mean = means[ling][fROI]
                    err = errs[ling][fROI]

                    _df = pd.DataFrame({
                        'contrast': names[19:21],
                        'estimate': mean[19:21],
                        'err': err[19:21]
                    })

                    if DUMP_TEXT:
                        _df.to_csv(prefix + 'plots/' + 'interaction.len.%s.baseline_%s.%s.txt' % (ling, baseline_key_str, fROI), index=False, sep='\t')

                    if PLOT_BAR:
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)
                        ax.tick_params(labelleft='on', labelbottom='on')
                        ax.yaxis.set_ticks_position('none')
                        ax.xaxis.set_ticks_position('bottom')
                        # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
                        ax.axhline(y=0, lw=1, c='gray', alpha=1)

                        colors_bylen_cur = colors_bylen[1:]

                        ax.bar(
                            r[19:21],
                            mean[19:21],
                            # color=[get_color(x, c) for x in names],
                            color=colors_bylen_cur,
                            edgecolor='none',
                            lw=1.5,
                            label=names[19:21],
                        )

                        for j, x in enumerate(names[19:21]):
                            ax.errorbar(
                                r[j + 19:j + 20],
                                mean[j + 19:j + 20],
                                yerr=err[j + 19:j + 20],
                                fmt='none',
                                ecolor='black',
                                lw=2,
                                capthick=capthick,
                                capsize=capsize
                            )

                        # plt.xlabel('Effect')
                        ax.set_xticks(r_base[19:21] * x_width)
                        if i == 5:
                            ax.set_xticklabels(names[19:21], rotation=rotation, ha='right')
                        else:
                            ax.set_xticklabels([])
                        if fROI == 'Mean':
                            ax.set_title(fROI)
                        else:
                            ax.set_title(fROI[4:])

                if PLOT_BAR:
                    # if baseline_key != 'none':
                    #     fig.suptitle('Exp %s, %s Controlled' % (experiment, ling_names.get(baseline_key, baseline_key)))
                    fig.set_size_inches(3, 12)
                    fig.tight_layout()

                    plt.savefig(prefix + 'plots/' + 'interaction.len.%s.baseline_%s.png' % (ling, baseline_key_str))
                plt.close('all')


    # Baseline effects

    if PLOT_BAR or DUMP_TEXT:
        fig = plt.figure()
        axes = []
        for i in range(6):
            axes.append((fROIs[i], fig.add_subplot(6, 1, i + 1)))

        for i, (fROI, ax) in enumerate(axes):
            pred_keys = sorted(baseline_means[fROI].keys())
            _mean = [baseline_means[fROI][x] for x in pred_keys]
            _err = [baseline_errs[fROI][x] for x in pred_keys]
            _names = [ling_names[x] for x in pred_keys]
            _r = np.arange(len(pred_keys)) * x_width

            _df = pd.DataFrame({
                'contrast': _names,
                'estimate': _mean,
                'err': _err
            })

            if DUMP_TEXT:
                _df.to_csv(prefix + 'plots/' + 'baseline_estimates.%s.txt' % fROI, index=False, sep='\t')

            if PLOT_BAR:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.tick_params(labelleft='on', labelbottom='on')
                ax.yaxis.set_ticks_position('none')
                ax.xaxis.set_ticks_position('bottom')
                # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
                ax.axhline(y=0, lw=1, c='gray', alpha=1)

                cmap = plt.get_cmap('gist_rainbow')

                ax.bar(
                    _r,
                    _mean,
                    color=[cmap(float(j) / clip) for j in range(clip)],
                    edgecolor='none',
                    lw=1.5,
                    label=_names,
                )

                for j, x in enumerate(_names):
                    ax.errorbar(
                        _r[j:j + 1],
                        _mean[j:j + 1],
                        yerr=_err[j:j + 1],
                        fmt='none',
                        ecolor='black',
                        lw=2,
                        capthick=capthick,
                        capsize=capsize
                    )

                # plt.xlabel('Effect')
                ax.set_xticks(_r)
                if i == 5:
                    ax.set_xticklabels(_names, rotation=rotation, ha='right')
                else:
                    ax.set_xticklabels([])
                if fROI == 'Mean':
                    ax.set_title(fROI)
                else:
                    ax.set_title(fROI[4:])

        if PLOT_BAR:
            fig.set_size_inches(6, 12)
            fig.tight_layout()

            plt.savefig(prefix + 'plots/' + 'baseline_estimates.png')
        plt.close('all')



    # Line plots

    for experiment in plot_data:
        for baseline in plot_data[experiment]:
            for ling in plot_data[experiment][baseline]:
                _df = []
                for x in plot_data[experiment][baseline][ling]:
                    __df = plot_data[experiment][baseline][ling][x]
                    __df['experiment'] = experiment
                    __df['baseline'] = baseline
                    __df['fROI'] = x
                    _df.append(__df)
                _df = pd.concat(_df, axis=0)
                _df['length'] = _df.contrast.map(cond2len)
                _df['stimtype'] = _df.contrast.map(cond2type)
                plot_data[experiment][baseline][ling] = _df

    if PLOT_LINES:
        # Real-words vs Jabberwocky

        fROIs.append('all')

        tick_labels = [
            'c01',
            'c02',
            'c03',
            'c04',
            'c06',
            'c12',
        ]

        # Exp 1

        fig = plt.figure(figsize=((12.5, 7)))
        axes = []
        for i in range(12):
            h = [Size.Fixed(0.5 + (i % 6) * 2), Size.Fixed(1.4)]
            v = [Size.Fixed(4 - (i // 6) * 3), Size.Fixed(2.5)]
            divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
            ax = fig.add_axes(
                divider.get_position(),
                axes_locator=divider.new_locator(nx=1, ny=1)
            )
            axes.append((fROIs[i % 6], ax))

        for i, (fROI, ax) in enumerate(axes[:6]):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.tick_params(labelleft='on', labelbottom='on')
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('none')
            # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
            ax.axhline(y=0, lw=1, c='gray', alpha=1)

            xticks = plot_data['2']['none']['none'].contrast
            xtickpos = np.arange(len(xticks))

            x = np.array([0, 1, 3, 4, 5])
            y = plot_data['1']['none']['none']
            y = y[(y.stimtype == 'c') & (y.fROI == fROI)]
            estimate = y.estimate.values
            err = y.err.values
            b = np.linalg.lstsq(np.stack([np.ones_like(x), x], axis=1), estimate)[0]
            xline = np.linspace(0, 5, 500)
            X = np.stack([np.ones_like(xline), xline], axis=1)
            yline = np.dot(X, b)

            ax.errorbar(
                x,
                estimate,
                yerr=err,
                fmt='ro',
                linestyle='none',
                ecolor='red',
                lw=2,
                capsize=0,
                label='normal'
            )
            ax.plot(
                xline,
                yline,
                linestyle='dashed',
                color='red'
            )
            ax.set_xticks([])
            # ax.set_title(fROI[4:])
            ax.set_ylim((-0.2, 1))

        for i, (fROI, ax) in enumerate(axes[6:]):
            i = i + 6
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.tick_params(labelleft='on', labelbottom='on')
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
            ax.axhline(y=0, lw=1, c='gray', alpha=1)

            xticks = plot_data['2']['none']['none'].contrast
            xtickpos = np.arange(len(xticks))

            x = np.arange(6)
            y = plot_data['2']['none']['none']
            y = y[(y.stimtype == 'c') & (y.fROI == fROI)]
            estimate = y.estimate.values
            err = y.err.values
            b = np.linalg.lstsq(np.stack([np.ones_like(x), x], axis=1), estimate)[0]
            xline = np.linspace(0, 5, 500)
            X = np.stack([np.ones_like(xline), xline], axis=1)
            yline = np.dot(X, b)

            ax.errorbar(
                x,
                estimate,
                yerr=err,
                fmt='ro',
                linestyle='none',
                ecolor='red',
                lw=2,
                capsize=0,
                label='normal'
            )
            ax.plot(
                xline,
                yline,
                linestyle='dashed',
                color='red'
            )

            x = np.array([0, 3, 5])
            y = plot_data['2']['none']['none']
            y = y[(y.stimtype == 'jab') & (y.fROI == fROI)]
            estimate = y.estimate.values
            err = y.err.values
            b = np.linalg.lstsq(np.stack([np.ones_like(x), x], axis=1), estimate)[0]
            xline = np.linspace(0, 5, 500)
            X = np.stack([np.ones_like(xline), xline], axis=1)
            yline = np.dot(X, b)

            ax.errorbar(
                x,
                estimate,
                yerr=err,
                fmt='bs',
                linestyle='none',
                fillstyle='none',
                ecolor='blue',
                lw=2,
                capsize=0,
                label='jabber'
            )
            ax.plot(
                xline,
                yline,
                linestyle='dashed',
                color='blue'
            )

            if i == 8:
                # get handles
                handles, labels = ax.get_legend_handles_labels()
                # remove the errorbars
                handles = [h[0] for h in handles]
                # use them in the legend
                ax.legend(handles, labels, loc='lower center', numpoints=1, frameon=False, bbox_to_anchor=(1.1, -0.35), ncol=2)

            ax.set_xticks(np.arange(6))
            ax.set_xticklabels(tick_labels, rotation=rotation, ha='right')
            ax.set_ylim((-0.2, 1))

        if not os.path.exists('output/conlen/plots'):
            os.makedirs('output/conlen/plots')
        plt.savefig('output/conlen/plots/all.png')

        plt.close('all')

        # Real-words vs non-constituents

        tick_labels = [
            'c03',
            'c04'
        ]

        fig = plt.figure(figsize=((12.5, 4)))
        axes = []
        for i in range(6):
            h = [Size.Fixed(0.5 + (i % 6) * 2), Size.Fixed(1.4)]
            v = [Size.Fixed(1), Size.Fixed(2.5)]
            divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
            ax = fig.add_axes(
                divider.get_position(),
                axes_locator=divider.new_locator(nx=1, ny=1)
            )
            axes.append((fROIs[i % 6], ax))

        for i, (fROI, ax) in enumerate(axes):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.tick_params(labelleft='on', labelbottom='on')
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
            ax.axhline(y=0, lw=1, c='gray', alpha=1)

            conditions = [
                'c03',
                'c04',
            ]

            x = np.arange(2)
            y = plot_data['2']['none']['none']
            y = y[y.contrast.isin(conditions) & (y.fROI == fROI)]
            estimate = y.estimate.values
            err = y.err.values
            b = np.linalg.lstsq(np.stack([np.ones_like(x), x], axis=1), estimate)[0]
            xline = np.linspace(0, 1, 500)
            X = np.stack([np.ones_like(xline), xline], axis=1)
            yline = np.dot(X, b)

            ax.errorbar(
                x,
                estimate,
                yerr=err,
                fmt='ro',
                linestyle='none',
                ecolor='red',
                lw=2,
                capsize=0,
                label='constituent'
            )
            ax.plot(
                xline,
                yline,
                linestyle='dashed',
                color='red'
            )

            conditions = [
                'nc03',
                'nc04',
            ]

            y = plot_data['2']['none']['none']
            y = y[y.contrast.isin(conditions) & (y.fROI == fROI)]
            estimate = y.estimate.values
            err = y.err.values
            b = np.linalg.lstsq(np.stack([np.ones_like(x), x], axis=1), estimate)[0]
            xline = np.linspace(0, 1, 500)
            X = np.stack([np.ones_like(xline), xline], axis=1)
            yline = np.dot(X, b)

            ax.errorbar(
                x,
                estimate,
                yerr=err,
                fmt='mx',
                linestyle='none',
                fillstyle='none',
                ecolor='m',
                lw=2,
                capsize=0,
                label='non-constituent'
            )
            ax.plot(
                xline,
                yline,
                linestyle='dashed',
                color='m'
            )

            if i == 2:
                # get handles
                handles, labels = ax.get_legend_handles_labels()
                # remove the errorbars
                handles = [h[0] for h in handles]
                # use them in the legend
                ax.legend(handles, labels, loc='lower center', numpoints=1, frameon=False, bbox_to_anchor=(1.1, -0.35),
                          ncol=2)

            ax.set_xticks(np.arange(2))
            ax.set_xticklabels(tick_labels, rotation=rotation, ha='right')
            ax.set_ylim((-0.2, 1))

        if not os.path.exists('output/conlen/plots'):
            os.makedirs('output/conlen/plots')
        plt.savefig('output/conlen/plots/nc.png')

        plt.close('all')



    # Interactions

    # Main result

    fig = plt.figure(figsize=((14, 4.5)))
    h = [Size.Fixed(0.7), Size.Fixed(13.3)]
    v = [Size.Fixed(0.5), Size.Fixed(3.7)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=1, ny=1)
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(labelleft='on', labelbottom='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3, zorder=1)
    # ax.axhline(y=0, lw=1, c='gray', alpha=1, zorder=2)

    contrasts = ['c-len', 'jab-len', 'c > jab', 'c-len > jab-len']
    contrasts_renamed = [''] * 5

    r_base = np.arange(len(contrasts_renamed))
    bar_width = 1./6 * 0.5
    cmap = plt.get_cmap('viridis')
    hatches = [
        None,
        '\\\\\\\\',
        '||||',
        '----',
        '....',
        'xxxx',
        '++++'
    ]

    for i, fROI in enumerate(['all'] + fROIs[:6]):
        r = r_base + i * bar_width
        df = []
        _df = plot_data['1']['none']['none']
        df.append(_df[(_df.contrast.isin(contrasts)) & (_df.fROI == fROI)])
        _df = plot_data['2']['none']['none']
        df.append(_df[(_df.contrast.isin(contrasts)) & (_df.fROI == fROI)])
        df = pd.concat(df, axis=0)
        df.contrast = contrasts_renamed
        hatch = hatches[i]

        ax.bar(
            r,
            df.estimate.values,
            color='w',
            edgecolor=(0.6, 0.6, 0.6),
            width=bar_width,
            label='Overall' if fROI == 'all' else fROI[4:],
            hatch=hatch,
            linewidth=2
        )

        ax.errorbar(
            r,
            df.estimate.values,
            yerr=df.err.values,
            fmt='none',
            ecolor=(0.8, 0.8, 0.8),
            capsize=4,
            capthick=2,
            linewidth=2
        )

    ax.legend(loc='upper right', ncol=2)

    ax.set_xticks(r_base + 0.25)
    ax.set_xticklabels(contrasts_renamed, rotation=rotation, ha='right')
    # ax.set_ylabel('BOLD')
    ax.set_ylim(0, 7)

    plt.savefig('output/conlen/plots/overall.png')

    # Non-constituents

    fig = plt.figure(figsize=((11, 4.5)))
    h = [Size.Fixed(0.7), Size.Fixed(10.3)]
    v = [Size.Fixed(0.5), Size.Fixed(3.7)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=1, ny=1)
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.tick_params(labelleft='on', labelbottom='off')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3, zorder=1)
    ax.axhline(y=0, lw=1, c='gray', alpha=1, zorder=2)

    contrasts = ['c-len-3-4', 'nc-len', 'c > nc', 'c-len > nc-len']
    contrasts_renamed = [''] * 4

    r_base = np.arange(len(contrasts_renamed))
    bar_width = 1./6 * 0.5
    cmap = plt.get_cmap('viridis')
    hatches = [
        None,
        '\\\\\\\\',
        '||||',
        '----',
        '....',
        'xxxx',
        '++++'
    ]

    for i, fROI in enumerate(['all'] + fROIs[:6]):
        r = r_base + i * bar_width
        df = []
        _df = plot_data['2']['none']['none']
        df.append(_df[(_df.contrast.isin(contrasts)) & (_df.fROI == fROI)])
        df = pd.concat(df, axis=0)
        df.contrast = contrasts_renamed
        hatch = hatches[i]

        ax.bar(
            r,
            df.estimate.values,
            color='w',
            edgecolor=(0.6, 0.6, 0.6),
            width=bar_width,
            label='Overall' if fROI == 'all' else fROI[4:],
            hatch=hatch,
            linewidth=2
        )

        ax.errorbar(
            r,
            df.estimate.values,
            yerr=df.err.values,
            fmt='none',
            ecolor=(0.8, 0.8, 0.8),
            capsize=4,
            capthick=2,
            linewidth=2
        )

    ax.legend(loc='upper right', ncol=4)

    ax.set_xticks(r_base + 0.25)
    ax.set_xticklabels(contrasts_renamed, rotation=rotation, ha='right')
    # ax.set_ylabel('BOLD')
    ax.set_ylim(-1, 6)

    plt.savefig('output/conlen/plots/nc-overall.png')
