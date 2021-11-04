import pandas as pd

df = pd.read_csv('timecourses/results/Nlength_con1/predictors/Constituents_Sept19-2016.csv')
df = df.to_dict('index')

itemmeasures = []
sentid = 0
for i in range(len(df)):
    itemnum = df[i]['Num']
    text = df[i]['TEXT']
    conlen = df[i]['COND']

    if conlen == 12:
        condcode = 'A'
    elif conlen == 6:
        condcode = 'B'
    elif conlen == '4':
        condcode = 'C'
    elif conlen == '2':
        condcode = 'G'
    elif conlen == '1':
        condcode = 'H'

    itemid = '%s%s' % (condcode, conlen)
    docid = '%s%s' % (condcode, itemnum)

    for j, w in enumerate(text.split()):
        sentpos = j + 1
        sentid = sentid + j // conlen

        itemmeasures.append((sentid, w, 'C', docid, sentpos, condcode, itemnum, conlen, itemid))

    sentid += 1

itemmeasures = pd.DataFrame(
    itemmeasures,
    columns=['sentid', 'word', 'cond', 'docid', 'sentpos', 'condcode', 'itemnum', 'conlen', 'itemid']
)
itemmeasures.to_csv('word-by-word_measures/conlen1_src.itemmeasures', sep=' ', index=False, na_rep='NaN')
