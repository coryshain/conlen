function extract_ts(expt_name)
    info=readtable([expt_name '.csv'], 'Delimiter',',');
    fields=info.Properties.VariableNames;
    for row = 1 : size(info,1) 
        subj=info(row,1); runs=info(row,2:end);
        uid=subj.(fields{1}){1}; uid=uid(1:3);
        f=fullfile(expt_name,subj.(fields{1}){1},'func');
        for run = 1 : numel(runs)
            fout=fullfile('results',sprintf('%s_%s_run%d.mat',expt_name,uid,run));
            nii=glob(fullfile(f,sprintf('%d',runs.(fields{run+1})), ...
                'results','preprocessing','*nii'));
            rois=glob(fullfile('ROIs',[subj.(fields{1}){1} '*.mat']));
            if ~isempty(nii)
                intersect_and_save(fout,nii{1},rois{1});
            end
        end
    end
end

function intersect_and_save(fout,nii,rois)
    time_courses=struct();
    addpath('/om/group/evlab/software/spm12');
    load(rois,'roiInds');
    niiData=spm_read_vols(spm_vol(nii));
    
    %language (LH)
    for lh = 1 : 6
        inds=roiInds.language.from90to100prcnt.all(lh);
        [x,y,z]=ind2sub([91 109 91],inds{1});
        for i = 1 : numel(x)
            time_courses.(sprintf('language_region%d',lh))(i,:)=niiData(x(i),y(i),z(i),:);
        end
    end
    %md
    for md = 1 : 20
        inds=roiInds.mult_demand.from90to100prcnt.all(md);
        [x,y,z]=ind2sub([91 109 91],inds{1});
        for i = 1 : numel(x)
            time_courses.(sprintf('md_region%d',md))(i,:)=niiData(x(i),y(i),z(i),:);
        end
    end
    save(fout,'time_courses');
end