import gl

fROIs = [
    'all',
    'LIFGorb',
    'LIFG',
    'LMFG',
    'LAntTemp',
    'LPostTemp',
    'LAngG'
]

PDD = [
    None,
    'IFGorb',
    'IFGtri',
    None,
    'TP-aSTS',
    'pSTS',
    'TPJ'
]

for f, p in zip(fROIs, PDD):
    # LANG only
    gl.resetdefaults()
    gl.azimuthelevation(-90, 15)
    gl.meshload('BrainMesh_ICBM152.mz3')

    gl.overlayload('parcels/FED_%s.nii' % f)
    gl.overlaycolorname(1, 'Black')
    gl.overlayminmax(1, 0.01, 10.0)

    gl.overlayload('parcels/FED_%s.nii' % f)
    gl.overlaycolorname(2, 'Black')
    gl.overlayminmax(2, 0.01, 10.0)

    gl.colorbarvisible(0)
    gl.overlaytransparencyonbackground(25)
    gl.meshcurv()
    gl.savebmp('output/conlen/plots/LANG_%s_parcels.png' % f)

    # LANG-PDD overlay
    if f not in ('all', 'LMFG'):
        gl.resetdefaults()
        gl.azimuthelevation(-90, 15)
        gl.meshload('BrainMesh_ICBM152.mz3')

        gl.overlayload('parcels/FED_%s.nii' % f)
        gl.overlaycolorname(1, 'Red')

        if p is not None:
            gl.overlayload('parcels/PDD_%s.nii' % p)
            gl.overlaycolorname(2, 'Blue')

        gl.colorbarvisible(0)
        gl.overlaytransparencyonbackground(25)
        gl.meshcurv()
        gl.savebmp('output/conlen/plots/LANG_PDD_%s_parcels.png' % f)

