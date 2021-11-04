parcel_path = '/home/cshain/cshain/pddparcels/';
parcel_names = {'IFGorb.nii',...
                'IFGtri.nii',...
                'TP.nii',...
                'aSTS.nii',...
                'pSTS.nii',...
                'TPJ.nii'};
parcels = {};
for i=1:length(parcel_names)
    parcel = dir(parcel_names{i});
    parcels{i} = fullfile(parcel_path,parcel.name);
end
vol = uint8(zeros(91,109,91));
for i=1:length(parcels)
    header = niftiinfo(parcels{i});
    vol = vol + (i*niftiread(header));
end
vol(vol==4)=3;
vol(vol==5)=4;
vol(vol==6)=5;

% ROI order: IFGorb, IFGtri, TP+aSTS, pSTS, TPJ

% Save merged ROIs
niftiwrite(vol, 'pdd_parcels.nii', header);

% Save TP-aSTS
out_names = {'PDD_IFGorb.nii',...
             'PDD_IFGtri.nii',...
             'PDD_TP-aSTS.nii',...
             'PDD_pSTS.nii',...
             'PDD_TPJ.nii'};

for i=1:5
    out = uint8(vol == i);
    niftiwrite(out, out_names{i}, header);
end

% Save unmerged Fedorenko parcels
fed_parcels_path = '/mindhive/evlab/u/Shared/ROIS_Nov2020/Func_Lang_LHRH_SN220/allParcels_language.nii';
header = niftiinfo(fed_parcels_path);
fed_parcels = niftiread(header);
parcel_names = {'LIFGorb', 'LIFG', 'LMFG', 'LAntTemp', 'LPostTemp', 'LAngG', ...
                'RIFGorb', 'RIFG', 'RMFG', 'RAntTemp', 'RPostTemp', 'RAngG'};

for i=1:6
    name = parcel_names{i};
    vol = single(fed_parcels == i);
    niftiwrite(vol, [parcel_path, 'FED_', name, '.nii'], header);  
end

fed_parcels(fed_parcels==3) = 0;
fed_parcels(fed_parcels==4) = 3;
fed_parcels(fed_parcels==5) = 4;
fed_parcels(fed_parcels==6) = 5;
fed_parcels(fed_parcels > 5) = 0;

niftiwrite(fed_parcels, [parcel_path, 'fed_parcels.nii'], header);

