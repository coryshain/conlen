import os
import re
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
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

    PLOT_LING = False
    PLOT_BAR = False

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


    # Items

    itemmeasures = pd.read_csv('output/conlen/nlength_con2/conlen2.itemmeasures', sep=' ')
    preds_means = itemmeasures.groupby(['cond', 'conlen', 'docid']).mean().reset_index()
    preds_mean = itemmeasures.groupby(['cond', 'conlen']).mean()
    preds_err = itemmeasures.groupby(['cond', 'conlen']).sem()


    plot_data = {}

    # Experiment 1

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
                plot_data[experiment][baseline_key_str][ling][fROI] = _df
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
            contrasts_cur = contrasts_cur[~contrasts_cur.contrast.isin(['isC34', 'isC1412', 'CLen34', 'CLen1412'])]
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

        if baseline_key != 'none':
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            mean = preds_mean.loc[:, baseline_key]
            err = preds_err.loc[:, baseline_key]

            _df = pd.DataFrame({
                'contrast': names[:clip],
                'estimate': mean[:clip],
                'err': err[:clip],
            })
            _df.to_csv(prefix + 'plots/' + 'items.%s.txt' % baseline_key, index=False, sep='\t')

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
                    r[:clip],
                    mean[:clip],
                    # color=[get_color(x, c) for x in names],
                    color=colors_bycond,
                    edgecolor='none',
                    lw=1.5,
                    label=names[:clip],
                )

                for j, x in enumerate(names[:clip]):
                    ax.errorbar(
                        r[j:j + 1],
                        mean[j:j + 1],
                        yerr=err[j:j + 1],
                        fmt='none',
                        ecolor='black',
                        lw=2,
                        capthick=capthick,
                        capsize=capsize
                    )

                # plt.xlabel('Effect')
                ax.set_xticks(r_base[:clip] * x_width)
                ax.set_xticklabels(names[:clip], rotation=rotation, ha='right')
                ax.set_ylabel(ling_names.get(baseline_key, baseline_key))

                fig.set_size_inches(clip, 4)
                fig.tight_layout()

                if not os.path.exists(prefix + 'plots'):
                    os.makedirs(prefix + 'plots')

                plt.savefig(prefix + 'plots/' + 'items.%s.png' % baseline_key)
            plt.close('all')

        for ling in means:
            fig = plt.figure()
            axes = []
            for i in range(6):
                axes.append((fROIs[i], fig.add_subplot(6, 1, i + 1)))

            for i, (fROI, ax) in enumerate(axes):
                mean = means[ling][fROI]
                err = errs[ling][fROI]

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

                plot_data[experiment][baseline_key_str][ling][fROI] = _df
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

                print(_df)


    # Line plots

    # frois = [
    #     'LANGLMFG',
    #     'LANGLAntTemp',
    #     'LANGLPostTemp',
    #     'LANGLAngG',
    #     'LANGLIFGorb',
    #     'LANGLIFG'
    # ]

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

    fig = plt.figure()
    axes = []
    axes.append((fROIs[i], fig.add_subplot(3, 3, i + 1)))
    for i in range(6):
        axes.append((fROIs[i], fig.add_subplot(3, 3, i + 1)))

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
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(tick_labels, rotation=rotation, ha='right')
        ax.set_title(fROI[4:])

        if i == 0:
            # get handles
            handles, labels = ax.get_legend_handles_labels()
            # remove the errorbars
            handles = [h[0] for h in handles]
            # use them in the legend
            ax.legend(handles, labels, loc='lower right', numpoints=1, frameon=False)

    fig.set_size_inches(12, 8)
    fig.tight_layout()

    if not os.path.exists('output/conlen/plots'):
        os.makedirs('output/conlen/plots')
    plt.savefig('output/conlen/plots/exp1.png')

    plt.close('all')

    # Exp 2

    fig = plt.figure()
    axes = []
    for i in range(6):
        axes.append((fROIs[i], fig.add_subplot(2, 3, i + 1)))

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

        x = np.array([2, 3])
        y = plot_data['2']['none']['none']
        y = y[(y.stimtype == 'nc') & (y.fROI == fROI)]
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
            fmt='gx',
            linestyle='none',
            ecolor='g',
            lw=2,
            capsize=0,
            label='non-const'
        )
        ax.plot(
            xline,
            yline,
            linestyle='dashed',
            color='g'
        )

        if i == 0:
            # get handles
            handles, labels = ax.get_legend_handles_labels()
            # remove the errorbars
            handles = [h[0] for h in handles]
            # use them in the legend
            ax.legend(handles, labels, loc='lower right', numpoints=1, frameon=False)

        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(tick_labels, rotation=rotation, ha='right')
        ax.set_title(fROI[4:])

    fig.set_size_inches(12, 8)
    fig.tight_layout()

    if not os.path.exists('output/conlen/plots'):
        os.makedirs('output/conlen/plots')
    plt.savefig('output/conlen/plots/exp2.png')

    plt.close('all')


