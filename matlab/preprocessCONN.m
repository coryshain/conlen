function preprocessCONN_Affourtit(expt_dir,segment)

addpath('/mindhive/evlab/u/iblank/Desktop/toolboxes/conn_toolbox_version_14');
addpath('/om/group/evlab/software/spm12');
parcelPath = '/mindhive/evlab/u/apaunov/Projects/MNI_Parcels/language/01_LH_IFGorb.nii';
parcelNames = {'Language01_LH_IFGorb'};

ssStruct=struct();
nSubj=0;
for f = dir(expt_dir)
    if ~strcmp(f.name(1),'.')
        nSubj=nSubj+1;
        ssStruct(nSubj).ssID=fullfile(f.folder,f.name);
        anat=glob(fullfile(f.folder,f.name,'anat','wc*nii'));

        if ~isempty(anat) && segment
            load('Segment_SPM12.mat','matlabbatch');
            matlabbatch{1}.spm.spatial.preproc.channel.vols{1} = anat{1};
            spm_jobman('run',matlabbatch);
            clear('matlabbatch');
        end
    %add to structure
    c1=glob(fullfile(f.folder,f.name,'anat','c1wc*nii'));
    c2=glob(fullfile(f.folder,f.name,'anat','c2wc*nii'));
    c3=glob(fullfile(f.folder,f.name,'anat','c3wc*nii'));
    ssStruct(nSubj).normalized=anat{1};
    ssStruct(nSubj).segmented{1}=c1{1};
    ssStruct(nSubj).segmented{2}=c2{1};
    ssStruct(nSubj).segmented{3}=c3{1};  
    end
end

save('ssStruct.mat','ssStruct');
%% Run CONN for each functional scan of each subject %%
for ss = 1 : numel(ssStruct)   
    functionals=glob(fullfile(ssStruct(ss).ssID,'func','swr*nii'));
    covariates=glob(fullfile(ssStruct(ss).ssID,'motion','rp*txt'));
    
    for funcInd = 1 : numel(functionals)
        tmp=split(functionals{funcInd},'/'); tmp=split(tmp{end},'.');
        tmp=split(tmp{1},'-'); run_num=tmp{end};
        
        clear('BATCH');
        BATCH.filename = fullfile(ssStruct(ss).ssID,'func',run_num);
        %% Setup %%
        BATCH.Setup.RT = 2.0;
        BATCH.Setup.normalized = true;

        BATCH.Setup.structurals{1} = ssStruct(ss).normalized;
        BATCH.Setup.masks.Grey.files{1} = ssStruct(ss).segmented{1};
        BATCH.Setup.masks.White.files{1} = ssStruct(ss).segmented{2};
        BATCH.Setup.masks.CSF.files{1} = ssStruct(ss).segmented{3};

        BATCH.Setup.functionals{1}{1} = functionals{funcInd};
        BATCH.Setup.masks.Grey.dimensions = 1;      % extract 1 principal component (mean) - default value
        BATCH.Setup.masks.White.dimensions = 16;    % extract 16 principal components  - default value
        BATCH.Setup.masks.CSF.dimensions = 16;      % extract 16 principal components  - default value

        BATCH.Setup.nsubjects = 1;
        BATCH.Setup.voxelmask = 2;                  % subject-specific mask
        BATCH.Setup.voxelresolution = 3;            % same resolution/registration as functional scans
        BATCH.Setup.analysisunits = 2;              % raw signal (not percent signal change)
        BATCH.Setup.outputfiles = [0,1,0,0,0,0];    % create confound-corrected nifti time-series
        BATCH.Setup.roiextract = 1;                 % use the functionals to extract ROI time-series

        for parcelInd = 1:length(parcelNames)
            BATCH.Setup.rois.names{parcelInd} = parcelNames{parcelInd};
            BATCH.Setup.rois.dimensions{parcelInd} = 1; 
            BATCH.Setup.rois.files{parcelInd}{1} = parcelPath;
            BATCH.Setup.rois.roiextract(parcelInd) = 0;
        end

        BATCH.Setup.conditions.names{1}=run_num;
        BATCH.Setup.conditions.onsets{1}{1}{1} = 0;
        BATCH.Setup.conditions.durations{1}{1}{1} = Inf; 

        BATCH.Setup.covariates.names{1} = 'realignment';   
        BATCH.Setup.covariates.files{1}{1}{1} = covariates{funcInd};

        BATCH.Setup.done = 1;                               % run the analysis
        BATCH.Setup.overwrite = 'Yes';
        BATCH.Setup.isnew = 1;

        %% Preprocessing %%
        BATCH.Preprocessing.filter = [0.0 Inf];
        BATCH.Preprocessing.confounds.names = ...
            {'White Matter'; 'CSF'; 'Effect'; 'realignment'};
        BATCH.Preprocessing.confounds.dimensions = {5; 5; 1; 6};    % default values
        BATCH.Preprocessing.confounds.deriv = {0; 0; 0; 1};         % default values
        BATCH.Preprocessing.detrending = 1;
        BATCH.Preprocessing.despiking = 0;
        BATCH.Preprocessing.done = 1;                               % run the analysis
        BATCH.Preprocessing.overwrite = 'Yes';

        conn_batch(BATCH)

        %% Delete unnecessary folders %%
        rmdir(fullfile(BATCH.filename, 'results', 'firstlevel'), 's');
        rmdir(fullfile(BATCH.filename, 'results', 'secondlevel'), 's');  
    end
end

   
end
