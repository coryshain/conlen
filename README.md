# fMRI_conlen

This repository contains code to reproduce the analyses in Shain et al. (in prep),
_Constituent length effects do not support syntactic abstraction in the
human language network_ from preprocessed fMRI timecourses to final statistical results.

The processing pipeline follows the following steps:
1. Extract tabular data from preprocessed fMRI timecourses
2. Regress first-level models to produce beta estimates for each participant
3. Compute contrast effects using linear combinations of beta estimates from first-level models
4. Regress mixed-effects second-level (group) models using first-level contrasts as the dependent variable
5. Statistically test second-level models

Each of these steps is supported by scripts in this codebase. In practice, SPM preprocessing and step 1 are skipped in
this readme and the outputs of step 1 are shared on OSF (see **Data** below).
## Installation

Install [anaconda](https://www.anaconda.com/), then run the following commands from this repository root to create and 
activate a new conda environment:

    conda env create -f conda_conlen.yml
    conda activate conlen
    
Components of this pipeline additionally require the `lme4` package in R. To install it, run the following commands:

    # Start an interactive R session
    R
    # From within the interactive session
    install.packages('lme4')
    install.packages('statmod')
    
The `conlen` environment must first be activated anytime you want to use this codebase:

    conda activate conlen


## Data

Data are not distributed with this repository but can be downloaded from [OSF](https://osf.io/7pknb/).
Once downloaded, all directories from OSF should be placed at the root of this repository.

We currently only distribute preprocessed tabular fMRI timecourses because the raw, intermediate, and final preprocessed
matlab files from SPM are quite large. However, we have provided the preprocessing scripts we used, and we can provide
source data upon request.

A matlab script that preprocesses the raw fMRI is provided in `matlab/preprocessCONN.m`, and a matlab script that
extracts functionally localized timecourses by fROI is provided in `matlab/extract_ts.m`. As written, these scripts
will not run out of the box because they are embedded in the EvLab file structure and software environment. The paths
in the scripts will need to be modified according to your system's file structure.

Tabular data were extracted from the preprocessed matlab files using `conlen/preprocess_conlen.py`.


## Core Usage

Regress first-level models as follows:

    python -m conlen.regress_l1
    
Compute contrasts from first-level models as follows:

    python -m conlen.run_contrasts
    
Regress second-level models as follows:

    Rscript conlen/regress_l2.R
    
Statistically test second-level models as follows:

    Rscript conlen/test_l2.R
    
Extract a significance table as follows:

    python -m conlen.signif_table
    
Apply FDR correction to the critical comparisons as follows:

    python -m conlen.fdr
    
All outputs will be dumped to a directory called `outputs` at the root of this repository.
The primary outputs are `output/signif_main.csv` and `output/signif_ling_diff.csv`, which respectively
contain FDR-corrected significance values for the critical comparisons and modulation of those comparisons
by linguistic variables.
    

## Additional Utilities

Generate statistics for the language localizer used in this study as follows:

    python -m conlen.localizer_stats

Generate plot panels such as those reported in the paper as follows:

    python -m conlen.plot
    
Compare our parcels with Pallier et al.'s as follows:

    python -m conlen.compare_parcels

The script `conlen/surfice.py` can be loaded into the software Surf Ice to generate cortical surface plots of 
parcels such as those shown in the paper and SI, but the paths within it will need to be modified to match your
system.