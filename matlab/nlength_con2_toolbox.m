
% set cfgs
localizer_expt = 'langlocSN';
critical_expt = 'Nlength_con2';
localizer_contrast = {'S-N'};
effect_contrasts = {'A_12c','B_6c','C_4c','E_3c','G_2c','H_1c','I_jab12c','J_jab4c','K_jab1c'};
fed_parcels = '/mindhive/evlab/u/Shared/ROIS_Nov2020/Func_Lang_LHRH_SN220/allParcels_language.nii';
pdd_parcels = '/path/to/parcels/pdd_parcels.nii';
outdir = fullfile(pwd,'fed');

% sessions
sessions = {...
    '252_FED_20181010b_3T2_PL2017',
    '524_FED_20180802a_3T2_PL2017',
    '674_FED_20180730b_3T2_PL2017',
    '676_FED_20180727a_3T2_PL2017',
    '678_FED_20180730a_3T2_PL2017',
    '679_FED_20180730c_3T2_PL2017',
    '680_FED_20180802b_3T2_PL2017',
    '681_FED_20180803a_3T2_PL2017',
    '682_FED_20180803b_3T2_PL2017',
    '683_FED_20180803c_3T2_PL2017',
    '684_FED_20180803d_3T2_PL2017',
    '685_FED_20180808a_3T2_PL2017',
    '688_FED_20180914a_3T2_PL2017',
    '689_FED_20181010a_3T2_PL2017',
    '691_FED_20181018a_3T2_PL2017',
    '692_FED_20181018b_3T2_PL2017',
    '693_FED_20181029a_3T2_PL2017',
    '694_FED_20181101a_3T2_PL2017',
    '695_FED_20181113a_3T2_PL2017',
    '696_FED_20181115a_3T2_PL2017',
    '697_FED_20181119a_3T2_PL2017',
    '698_FED_20181129a_3T2_PL2017',
    '700_FED_20181128a_3T2_PL2017',
    '702_FED_20181214a_3T2_PL2017',
    '704_FED_20190107a_3T2_PL2017',
    };
localizer_spmfiles = strcat('/mindhive/evlab/u/Shared/SUBJECTS/',sessions,'/firstlevel_',localizer_expt,'/SPM.mat');
critical_spmfiles = strcat('/mindhive/evlab/u/Shared/SUBJECTS/',sessions,'/firstlevel_',critical_expt,'/SPM.mat');

% toolbox control structure
ss=struct(...
    'swd',outdir,...
    'EffectOfInterest_spm',{critical_spmfiles},...
    'Localizer_spm',{localizer_spmfiles},...
    'EffectOfInterest_contrasts',{effect_contrasts},...
    'Localizer_contrasts',{localizer_contrast},...
    'Localizer_thr_type','percentile-ROI-level',...
    'Localizer_thr_p',0.1,...
    'type','mROI',...
    'ManualROIs',fed_parcels,...
    'overlap_thr_roi', 0,...
    'model',1,...
    'estimation','OLS',...
    'overwrite',true,...
    'ExplicitMasking','',...
    'ask', 'none'...
    );

% initialize and run toolbox
addpath('/om/group/evlab/software/conn');
conn_module('el','init');
ss=spm_ss_design(ss);
ss=spm_ss_estimate(ss);
