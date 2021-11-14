# fMRI_conlen

This repository contains code to reproduce the analyses in Shain et al. (in prep),
_'Constituent length' effects in fMRI do not provide evidence for abstract syntactic processing_ from preprocessed fMRI timecourses to final statistical results.

The processing pipeline follows the following steps:
1. (Optional) Extract tabular data from preprocessed fMRI timecourses
2. Regress first-level models to produce beta estimates for each participant
3. Compute contrast effects using linear combinations of beta estimates from first-level models
4. Regress mixed-effects second-level (group) models using first-level contrasts as the dependent variable
5. Statistically test second-level models

Each of these steps is supported by scripts in this codebase.

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

To protect participants' privacy, we only distribute preprocessed fMRI data (the outputs of our SPM preprocessing
pipeline). For transparency, we have included our preprocessing scripts in the `matlab` directory of this
repository. The script used to preprocess the raw fMRI is provided in `matlab/preprocessCONN.m`, and the script that
extracted functionally localized timecourses by fROI is provided in `matlab/extract_ts.m`.


## Core Usage

Compile a tabular design matrix from SPM preprocessing outputs and word-by-word linguistic predictors:

    python -m conlen.preprocess_conlen
    
This step can be time-consuming, so we have made it optional by including its outputs
(tabular data for regressions) on OSF, allowing users to skip straight to regression.

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
