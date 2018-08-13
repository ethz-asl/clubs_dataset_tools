function deleteBadImagesAndPoses (imagesUsedArray, imageLocationArray, posesFile, nCam) 
    poses = readtable(poses_file);

    usedImages = cat(2,imagesUsedArray{:});
    usedByAll = min(usedImages,[],2);
    notUsedIndices = find(usedByAll==0);
     
    for iCam = 1 : nCam
        allNames = imageLocationArray{iCam};
        namesNotUsed = allNames(notUsedIndices);
        for i = 1 : length(namesNotUsed)
            disp(namesNotUsed{i})
            delete(namesNotUsed{i})
        end
    end
    
    poses(notUsedIndices,:) = [];
    
    % TODO(ntonci): Remove variable name
    writetable(poses, posesFile);
end