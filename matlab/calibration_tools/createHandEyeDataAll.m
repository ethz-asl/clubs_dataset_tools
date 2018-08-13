function createHandEyeDataAll (posesPath, imagesUsed, cameraCalibration, saveFolder, nCam)
    % TODO(ntonci): Currently assumes that cameraCalibration contains only imagesUsed and not more.
    if (exist(saveFolder, 'dir') == 0), mkdir(saveFolder); end

    allUsedImages = cat(2,imagesUsed{:});
    catImagesUsed = min(allUsedImages,[],2);
    usedIndices = find(catImagesUsed>0);
    notUsedIndices = find(catImagesUsed==0);

    poses = csvread(posesPath);
    poses = poses(usedIndices,:);

    outPosesRobot = zeros(length(poses), 8);
    outPosesRobot(:,1) = poses(:,1); % timestamp
    outPosesRobot(:,2) = poses(:,5); % x
    outPosesRobot(:,3) = poses(:,9); % y
    outPosesRobot(:,4) = poses(:,13); % z

    for idx = 1 : length(poses)
        rotMatrix = [poses(idx,2:4); poses(idx,6:8); poses(idx,10:12)];
        quat = rotm2quat(rotMatrix');
        outPosesRobot(idx,5) = quat(2); % qx
        outPosesRobot(idx,6) = quat(3); % qy
        outPosesRobot(idx,7) = quat(4); % qz
        outPosesRobot(idx,8) = quat(1); % qw
    end

    dlmwrite([saveFolder 'robot_poses.csv'], outPosesRobot, 'delimiter', ',', 'precision', 15);

    indicesToRemove = {};
    for iCam = 1 : nCam
        indicesToRemove{iCam} = [];
        for idx = 1 : length(notUsedIndices)
            if (imagesUsed{iCam}(notUsedIndices(idx)) ~= 0)
                indicesToRemove{iCam} = [indicesToRemove{iCam}; sum(imagesUsed{iCam}(1:notUsedIndices(idx)))];
            end
        end
    end


    imageTranslations = {};
    imageRotations = {};
    for iCam = 1 : nCam
        imageTranslations{iCam} = cameraCalibration{iCam}.TranslationVectors;
        imageTranslations{iCam}(indicesToRemove{iCam},:) = [];
        imageRotations{iCam} = cameraCalibration{iCam}.RotationMatrices;
        imageRotations{iCam}(:,:,indicesToRemove{iCam}) = [];
    end


    for iCam = 1 : nCam
        outPosesCamera = zeros(length(poses), 8);
        outPosesCamera(:,1) = poses(:,1); % timestamp
        outPosesCamera(:,2:4) = imageTranslations{iCam}./1000; % x, y, z

        for idx = 1 : length(poses)
            rotMatrix = imageRotations{iCam}(:,:,idx);
            quat = rotm2quat(rotMatrix');
            outPosesCamera(idx,5) = quat(2); % qx
            outPosesCamera(idx,6) = quat(3); % qy
            outPosesCamera(idx,7) = quat(4); % qz
            outPosesCamera(idx,8) = quat(1); % qw
        end

        dlmwrite([saveFolder 'camera_' num2str(iCam) '_poses.csv'], outPosesCamera, ...
            'delimiter', ',', 'precision', 15);
    end

end
