#!/usr/bin/env Rscript

pr = function(x, buffer=NULL) {
    if (is.null(buffer)) {
        buffer = stderr()
    }
    cat(paste0(x, '\n'), file=buffer, append=TRUE)
}

library(lme4)

f_main_full = 'estimate ~ (1 | subject) + (1 | fROI)'
f_main_abl = 'estimate ~ 0 + (1 | subject) + (1 | fROI)'
f_main_full_fROI = 'estimate ~ 1'
f_main_abl_fROI = 'estimate ~ 0'
f_interaction_full = 'estimate ~ isCcond + (isCcond | subject) + (isCcond | fROI)'
f_interaction_abl = 'estimate ~ (isCcond | subject) + (isCcond | fROI)'
f_interaction_full_fROI = 'estimate ~ isCcond'
f_interaction_abl_fROI = 'estimate ~ 1'

for (experiment in c(1, 2)) {
    prefix_in = paste0('output/conlen/nlength_con', experiment, '/contrasts/')
    prefix_out = paste0('output/conlen/nlength_con', experiment, '/lme/')
    if (!dir.exists(prefix_out)) {
        dir.create(prefix_out, recursive = TRUE)
    }
    for (f in list.files(prefix_in)) {
        path = paste0(prefix_in, f)
        b = strsplit(f, '.', fixed=TRUE)[[1]][[2]]
        b_str = b
        df = read.table(path, header=TRUE, sep=',')
        isCcond = c(
            c('isC', 'isC34', 'isC1412', 'isCLen1', 'isCLen2', 'isCLen3', 'isCLen4', 'isCLen6', 'isCLen12', 'CLen', 'CLen34', 'CLen1412'),
            c('isCDiff', 'isC34Diff', 'isC1412Diff', 'isCLen1Diff', 'isCLen2Diff', 'isCLen3Diff', 'isCLen4Diff', 'isCLen6Diff', 'isCLen12Diff', 'CLenDiff', 'CLen34Diff', 'CLen1412Diff')
        )
        df$isCcond = FALSE
        df$isCcond[df$contrast %in% isCcond] = TRUE

        contrasts = c('CLen')
        if (b != 'none') {
            contrasts = c(contrasts, 'CLenDiff')
        }
        if (experiment == 2) {
            contrasts = c(contrasts, c('isC', 'isJAB', 'isNC', 'JABLen', 'NCLen', 'CLen34', 'CLen1412', 'C>JAB', 'C>NC', 'CLen>JABLen', 'CLen>NCLen'))
            if (b != 'none') {
                contrasts = c(contrasts, c('isCDiff', 'isJABDiff', 'isNCDiff', 'JABLenDiff', 'NCLenDiff', 'CLen34Diff', 'CLen1412Diff', 'C>JABDiff', 'C>NCDiff', 'CLen>JABLenDiff', 'CLen>NCLenDiff'))
            }
        }

        for (contrast in contrasts) {
            for (fROI in c('allfROI', levels(df$fROI))) {
                for (mtype in c('full', 'abl')) {
                    if (mtype == 'full') {
                        if (fROI == 'allfROI') {
                            mform = f_main_full
                        } else {
                            mform = f_main_full_fROI
                        }
                    } else { # mtype == 'abl
                        if (fROI == 'allfROI') {
                            mform = f_main_abl
                        } else {
                            mform = f_main_abl_fROI
                        }
                    }

                    if (b == 'none' | substr(b, 0, 3) == 'dlt') {
                        df_ = df[(df$contrast == contrast) & (df$ling == 'none'),]
                        if (fROI == 'allfROI') {
                            m = lmer(mform, REML=F, data=df_)
                        } else {
                            df_ = df_[df_$fROI == fROI,]
                            m = lm(mform, data=df_)
                        }

                        save(m, file=paste0(prefix_out, 'conlen.', b_str, '.', gsub('>', '_gt_', contrast), '.', fROI, '.', mtype, '.lme.Rdata'))

                        sink(paste0(prefix_out, 'conlen.', b_str, '.', gsub('>', '_gt_', contrast), '.', fROI, '.', mtype, '.lme.summary.txt'))
                        pr("Formula:", stdout())
                        pr(mform, stdout())
                        print(summary(m))
                        sink()
                    }
                }
            }
        }

        if (experiment == 2) {
            interactions = list(
                c('isC1412', 'isJAB'),
                c('isC34', 'isNC'),
                c('CLen1412', 'JABLen'),
                c('CLen34', 'NCLen')
            )

            for (interaction in interactions) {
                for (fROI in c('allfROI', levels(df$fROI))) {
                    for (mtype in c('full', 'abl')) {
                        c = interaction[[1]]
                        o = interaction[[2]]

                        if (mtype == 'full') {
                            if (fROI == 'allfROI') {
                                mform = f_interaction_full
                            } else {
                                mform = f_interaction_full_fROI
                            }
                        } else { # mtype == 'abl
                            if (fROI == 'allfROI') {
                                mform = f_interaction_abl
                            } else {
                                mform = f_interaction_abl_fROI
                            }
                        }

                        if (b == 'none' | substr(b, 0, 3) == 'dlt') {
                            df_ = df[(df$contrast %in% interaction) & (df$ling == 'none'),]
                            if (fROI == 'allfROI') {
                                m = lmer(mform, REML=F, data=df_)
                            } else {
                                df_ = df_[df_$fROI == fROI,]
                                m = lm(mform, data=df_)
                            }

                            save(m, file=paste0(prefix_out, 'conlen.', b, '.', c, '_v_', o, '.', fROI, '.', mtype, '.lme.Rdata'))

                            sink(paste0(prefix_out, 'conlen.', b, '.', c, '_v_', o, '.', fROI, '.', mtype, '.lme.summary.txt'))
                            pr("Formula:", stdout())
                            pr(mform, stdout())
                            print(summary(m))
                            sink()
                        }
                    }
                }
            }
        }
    }
}
