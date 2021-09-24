import sys
import os
import re
import argparse


# Thanks to Daniel Sparks on StackOverflow for this one (post available at
# http://stackoverflow.com/questions/5084743/how-to-print-pretty-string-output-in-python)
def pretty_table(row_collection, key_list, field_sep=' '):
  return '\n'.join([field_sep.join([str(row[col]).ljust(width)
    for (col, width) in zip(key_list, [max(map(len, column_vector))
      for column_vector in [ [v[k]
        for v in row_collection if k in v]
          for k in key_list ]])])
            for row in row_collection])

def csv_table(row_collection, key_list):
    return '\n'.join([','.join(z) for z in ([[str(x[y]) for y in key_list] for x in row_collection])])

def compute_row(path, stars=True):
    experiment = re.search('nlength_con(.)', path).group(1)
    baseline, contrast, fROI = path.split('.')[-6:-3]
    conv = 't'
    sing = 'f'
    beta = '---'
    se = '---'
    t = '---'
    p = '---'

    with open(path, 'r') as f:
        in_lrt = False
        in_full = False
        in_fixef = False
        for line in f:
            # print(line)
            # print(in_lrt)
            # print(in_full)
            # print(in_fixef)
            # print(beta, se, t)
            # raw_input()
            if 'isSingular' in line:
                sing = 't'
            if 'failed to converge' in line:
                conv = 'f'
            if not line.strip():
                if in_full and in_fixef:
                    in_full = False
                    in_fixef = False
            if line.startswith('Full model'):
                in_full = True
            elif line.startswith('Fixed effects:') or line.startswith('Coefficients:'):
                in_fixef = True
            elif line.startswith('(Intercept)'):
                if in_full and in_fixef:
                    _, beta, se, t = line.strip().split()[:4]
            elif line.startswith('isC'):
                if in_full and in_fixef:
                    _, beta, se, t = line.strip().split()[:4]
            elif line.strip().startswith('npar') or line.strip().startswith('Res.Df'):
                in_lrt = True
            elif line.startswith('m_full') or line.startswith('2'):
                if in_lrt:
                    line_parts = line.strip().split()
                    if len(line_parts) > 8:
                        p = float(line_parts[8])
                    elif len(line_parts) > 6:
                        p = float(line_parts[6])
                    else:
                        p = 1
                    if stars:
                        stars = 0
                        if p < 0.05:
                            stars += 1
                        if p < 0.01:
                            stars += 1
                        if p < 0.001:
                            stars += 1
                    p = '%.3e' % p
                    if stars:
                        p += ('*' * stars)
                    in_lrt = False

    return {
        'experiment': experiment,
        'baseline': baseline,
        'contrast': contrast,
        'fROI': fROI,
        'conv': conv,
        'sing': sing,
        'beta': beta,
        'se': se,
        't': t,
        'p': p
    }


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Get table of significance values for conlen tests.
    ''')
    argparser.add_argument('-c', '--csv', action='store_true', help='Print output as CSV. Otherwise pretty print')
    argparser.add_argument('-s', '--stars', action='store_true', help='Include significance stars')
    args = argparser.parse_args()
    cols = ['experiment', 'baseline', 'contrast', 'fROI', 'conv', 'sing', 'beta', 'se', 't', 'p']
    rows = []
    for experiment in ['1', '2']:
        directory = 'output/conlen/nlength_con%s/tests' % experiment
        paths = sorted([os.path.join(directory, x) for x in os.listdir(directory) if x.endswith('.lrt.summary.txt')])
        rows += [compute_row(path, stars=args.stars) for path in paths]

    rows = sorted(rows, key=lambda x: (x['baseline'], len(x['contrast']), x['contrast'].split('!')[0]))
    rows.insert(0, {c:c for c in cols})

    if args.csv:
        sys.stdout.write(csv_table(rows, cols) + '\n')
    else:
        sys.stdout.write(pretty_table(rows, cols) + '\n')
    sys.stdout.flush()

