function deleteBadImagesAndPoses (images_used_array, image_location_array, poses_file, nCam) 
    poses = readtable(poses_file);

    used_images = cat(2,images_used_array{:});
    used_by_all = min(used_images,[],2);
    not_used_indices = find(used_by_all==0);
     
    for iCam = 1 : nCam
        all_names = image_location_array{iCam};
        names_not_used = all_names(not_used_indices);
        for i = 1 : length(names_not_used)
            disp(names_not_used{i})
            delete(names_not_used{i})
        end
    end
    
    poses(not_used_indices,:) = [];
    
    % TODO(ntonci): Remove variable name
    writetable(poses, poses_file);
end