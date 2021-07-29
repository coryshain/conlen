#!/usr/bin/env Rscript

pr = function(x, buffer=NULL) {
    if (is.null(buffer)) {
        buffer = stderr()
    }
    cat(paste0(x, '\n'), file=buffer, append=TRUE)
}

library(lme4)


for (experiment in c(1, 2)) {
    prefix_in = paste0('output/conlen/nlength_con', experiment, '/lme/')
    prefix_out = paste0('output/conlen/nlength_con', experiment, '/tests/')
    if (!dir.exists(prefix_out)) {
        dir.create(prefix_out, recursive = TRUE)
    }

    models = list.files(path=prefix_in, pattern="*.Rdata", full.names=TRUE, recursive=FALSE)

    for (mpath in models) {
        if (grepl('full', mpath)) {
            path_parts = strsplit(mpath, '.', fixed=TRUE)[[1]]
            fROI = path_parts[[length(path_parts) - 3]]
            contrast = path_parts[[length(path_parts) - 4]]
            baseline = path_parts[[length(path_parts) - 5]]

            m_full_path = mpath
            m_abl_path = gsub('full', 'abl', mpath)

            m_full = get(load(m_full_path))
            m_abl = get(load(m_abl_path))
            lrt = anova(m_full, m_abl)

            sum_path = paste0(prefix_out, 'conlen.', baseline, '.', contrast, '.', fROI, '.lrt.summary.txt')

            sink(sum_path)
            pr('==================\nLikelihood ratio test\n', stdout())
            pr(paste0('Experiment: ', experiment), stdout())
            pr(paste0('Variable:   \n', contrast), stdout())
            pr(paste0('Baseline:   \n', baseline), stdout())
            pr(paste0('fROI:       \n', fROI), stdout())
            print(lrt)
            pr('------------------\nFull model\n', stdout())
            pr(paste0('Path:       ', m_full_path), stdout())
            print(summary(m_full))
            pr('\n\n', stdout())
            pr('------------------\nAblated model\n', stdout())
            pr(paste0('Path:       ', m_abl_path), stdout())
            print(summary(m_abl))
            sink()
        }
    }
}

