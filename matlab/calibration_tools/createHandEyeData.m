function createHandEyeData (posesPath, imagesUsed, cameraCalibration, saveFolder)
    % TODO(ntonci): Currently assumes that cameraCalibration contains only imagesUsed and not more.
    if (exist(saveFolder, 'dir') == 0), mkdir(saveFolder); end

    poses = csvread(posesPath);
    poses = poses(imagesUsed==1,:);

    outPosesRobot = zeros(length(poses), 8);
    outPosesRobot(:,1) = poses(:,1); % timestamp
    outPosesRobot(:,2) = poses(:,5); % x
    outPosesRobot(:,3) = poses(:,9); % y
    outPosesRobot(:,4) = poses(:,13); % z

    for idx = 1 : length(poses)
        rotMatrix = [poses(idx,2:4); poses(idx,6:8); poses(idx,10:12)];
        quat = rotm2quat(rotMatrix);
        outPosesRobot(idx,5) = quat(2); % qx
        outPosesRobot(idx,6) = quat(3); % qy
        outPosesRobot(idx,7) = quat(4); % qz
        outPosesRobot(idx,8) = quat(1); % qw
    end

    dlmwrite([saveFolder 'robot_poses.csv'], outPosesRobot, 'delimiter', ',', 'precision', 15);

    outPosesCamera = zeros(length(poses), 8);
    outPosesCamera(:,1) = poses(:,1); % timestamp
    outPosesCamera(:,2:4) = cameraCalibration.TranslationVectors./1000; % x, y, z

    for idx = 1 : length(poses)
        rotMatrix = cameraCalibration.RotationMatrices(:,:,idx);
        quat = rotm2quat(rotMatrix');
        outPosesCamera(idx,5) = quat(2); % qx
        outPosesCamera(idx,6) = quat(3); % qy
        outPosesCamera(idx,7) = quat(4); % qz
        outPosesCamera(idx,8) = quat(1); % qw
    end

    dlmwrite([saveFolder 'camera_poses.csv'], outPosesCamera, 'delimiter', ',', 'precision', 15);

end
