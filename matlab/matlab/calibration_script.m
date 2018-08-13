% Parameters
folders = {'../data/chameleon3/rgb_images'; ...
    '../data/primesense/rgb_images'; ...
    '../data/primesense/ir1_images'; ...
    '../data/realsense_d415/rgb_images'; ...
    '../data/realsense_d415/ir1_images'; ...
    '../data/realsense_d415/ir2_images'; ...
    '../data/realsense_d435/rgb_images'; ...
    '../data/realsense_d435/ir1_images'; ...
    '../data/realsense_d435/ir2_images'};
cameraNames = {'chameleon_rgb', ...
               'primesense_rgb', ...
               'primesense_ir', ...
               'd415_rgb', ...
               'd415_ir1', ...
               'd415_ir2', ...
               'd435_rgb', ...
               'd435_ir1', ...
               'd435_ir2'};
cameraSets = {'chameleon', ...
              'primesense', ...
              'realsense_d415', ...
              'realsense_d435'};
extrinsicsSet = [3, 2;  % ps_ir -> ps_rgb
                 6, 5;  % d415_ir2 -> d415_ir1
                 5, 4;  % d415_ir1 -> d415_rgb
                 9, 8;  % d435_ir2 -> d435_ir1
                 8, 7;  % d435_ir1 -> d435_rgb
                 1, 2;  % cham_rgb -> ps_rgb
                 4, 2;  % d415_rgb -> ps_rgb
                 7, 2]; % d435_rgb -> ps_rgb
exportSet = [1, 0, 0;  % cham
             2, 3, 0;  % ps
             4, 5, 6;  % d415
             7, 8, 9]; % d435

calibrationBoardSize = 50;
numberOfCameras = length(folders);
displayDetectedCorners = false;
displayReprojectionError = false;
displayExtrinsics = false;
removeOutliers = true;
increaseBrightnessOfIRImage = false;
shellPath = '~/.zshrc';
handEyeCalibrationWorkspace = '~/sandbox_catkin_ws/devel/setup.zsh';
handEyeCalibrationPath = '~/sandbox_catkin_ws/src/hand_eye_calibration/hand_eye_calibration/bin';


% Get all the image locations.
imageFolderArray = {};
imageLocationArray = {};
for iCam = 1: numberOfCameras
    imageFolderArray{iCam} = folders{iCam};
    images = imageSet(imageFolderArray{iCam});
    imageLocationArray{iCam} = images.ImageLocation;
end


% Increase brighntess of primesense IR image.
if increaseBrightnessOfIRImage
    disp('WARNING: Only increase brightness once in the beginning!')
    brightnessIncreaseFactor = 150;
    increaseBrightnessIRImages(imageLocationArray{3}, brightnessIncreaseFactor);
end


% Detect corners
imagePointsArray = {};
boardSizeArray = {};
imagesUsedArray = {};
imagesUsedLocation = {};
for iCam = 1 : numberOfCameras
    disp(['Detecting corners for camera ' num2str(iCam) '...']);

    [imagePointsArray{iCam}, boardSizeArray{iCam}, imagesUsedArray{iCam}] =  ...
        detectCorners (imageLocationArray{iCam}, displayDetectedCorners, iCam);
    imagesUsedLocation{iCam} = imageLocationArray{iCam}(find(imagesUsedArray{iCam}==1));
end
worldPoints = generateCheckerboardPoints(boardSizeArray{1}, calibrationBoardSize);


% Calibrate intrinsics for each camera individually.
calibrationParameters = {};
estimationErrors = {};
imagesToUse = {};

iteration = 0;
iCam = 1;
while iCam <= numberOfCameras
    disp(['Calibrating camera ' num2str(iCam) '...']);

    [calibrationParameters{iCam}, imagesToUse{iCam}, estimationErrors{iCam}] = ...
        calibrateIntrinsics(imagePointsArray{iCam}, worldPoints, displayReprojectionError, ...
        displayExtrinsics, removeOutliers, iCam);

    % Remove outliers
    maxIterations = 5;
    if sum(imagesToUse{iCam}==0) > 0
        disp(['Have outliers in iteration ' num2str(iteration) ', error is ' ...
            num2str(calibrationParameters{iCam}.MeanReprojectionError)]);

        idxToRemove = find(imagesToUse{iCam} == 0);

        idxToSet = [];
        for idx = 1 : length(idxToRemove)
            fistNElements = find(imagesUsedArray{iCam},idxToRemove(idx),'first');
            idxToSet = [idxToSet; fistNElements(end)];
        end
        imagesUsedArray{iCam}(idxToSet) = 0;
        imagePointsArray{iCam}(:,:,idxToRemove) = [];
        imagesUsedLocation{iCam}(idxToRemove) = [];
        iteration = iteration + 1;
        
        if iteration > maxIterations
            disp('Reached maximum number of iterations for outlier rejection.')
            iCam = iCam + 1;
            iteration = 0;
        end
    else
        disp(['No more outliers after iteration ' num2str(iteration) ', error is ' ...
            num2str(calibrationParameters{iCam}.MeanReprojectionError)]);
        iCam = iCam + 1;
        iteration = 0;
    end

end


% Loop over camera sets to calibrate extrinsics.
extrinsicsCalibrationParameters = {};
extrinsicsEstimationErrors = {};
extrinsicsImagesUsed = {};
for currentSet = 1 : size(extrinsicsSet, 1)
    iCams =  extrinsicsSet(currentSet, :);
    disp(['Calibrating camera extrinsics for cameras ' num2str(iCams(1)) ' and ' num2str(iCams(2)) '...']);

    [extrinsicsCalibrationParameters{currentSet}, ...
        extrinsicsEstimationErrors{currentSet}, extrinsicsImagesUsed{currentSet}] = ...
        calibrateExtrinsics (imagePointsArray{iCams(1)}, imagePointsArray{iCams(2)}, ...
        imagesUsedArray{iCams(1)}, imagesUsedArray{iCams(2)}, ...
        worldPoints, displayReprojectionError, displayExtrinsics, iCams);
end


% Output Hand-Eye calibration poses for individual camera.
for iCam = 1 : numberOfCameras
    disp(['Outputing hand-eye calibration data for camera ' num2str(iCam) ' ...']);
    createHandEyeData ('../data/poses.csv', imagesUsedArray{iCam}, ...
        calibrationParameters{iCam}, ['../results/cam' num2str(iCam) '/']);
end


% Output Hand-Eye calibration poses for all the cameras.
disp('Outputing hand-eye calibration data for all cameras ...');
createHandEyeDataAll ('../data/poses.csv', imagesUsedArray, calibrationParameters, ...
    '../results/all/', 9);

save('../results/calibrationParameters.mat', 'calibrationParameters');
save('../results/extrinsicsCalibrationParameters.mat', 'extrinsicsCalibrationParameters');


% Compute Hand-Eye calibration
for iCam = 1 : numberOfCameras
    disp(['Performing hand-eye calibration for camera ' num2str(iCam) ' ...']);
    path = '../results/cam';
    command = strcat({'source ./../scripts/hand_eye_calib.sh '}, shellPath, {' '}, ...
        handEyeCalibrationWorkspace, {' '}, handEyeCalibrationPath, {' '}, ...
        path, num2str(iCam), {'/'}, {'robot_poses.csv'}, {' '}, path, num2str(iCam), {'/'}, ...
        {'camera_poses.csv'}, {' '}, path, num2str(iCam), {'/'}, {'h_e_calib.json'});
    [status,cmdout] = system(command{1});
    location = strfind(cmdout,'Solution found with sample:');
    if (~isempty(location))
        cmdout = eraseBetween(cmdout,1,location-1);
    end
    location = strfind(cmdout,'Number of inliers:');
    disp(cmdout(location(end):location(end)+20));
    location = strfind(cmdout,'RMSE position:');
    disp(cmdout(location(end):location(end)+28));
    location = strfind(cmdout,'RMSE orientation:');
    disp(cmdout(location(end):location(end)+28));
    location = strfind(cmdout,'Translation norm:');
    disp(cmdout(location(end):location(end)+28));
end


% Export yaml files.
staticTransformID = fopen('../results/static_transform_extrinsics.yaml', 'w');
for currentSet = 1 : size(exportSet, 1)
    disp(['Writing out yaml files for ', cameraSets{currentSet}]);
    iCams =  exportSet(currentSet, :);
    fileID = fopen(['../results/' cameraSets{currentSet} '.yaml'],'w');
    for iCam = 1 : length(iCams)
        if (iCams(iCam) == 0)
            break;
        end

        if (iCam == 1)
            handEyeFileID = fopen(['../results/cam' num2str(iCams(iCam)) ...
            '/h_e_calib.json']);
            handEyeTransform = textscan(handEyeFileID,'%s');
            fclose(handEyeFileID);

            qx = str2double(handEyeTransform{1}{7}(1:end-1));
            qy = str2double(handEyeTransform{1}{9}(1:end-1));
            qz = str2double(handEyeTransform{1}{11}(1:end-1));
            qw = str2double(handEyeTransform{1}{13});
            x = str2double(handEyeTransform{1}{18}(1:end-1));
            y = str2double(handEyeTransform{1}{20}(1:end-1));
            z = str2double(handEyeTransform{1}{22});

            extrinsics = eye(4);
            extrinsics(1:3, 1:3) = quat2rotm([qw, qx, qy, qz]);
            extrinsics(1,4) = x;
            extrinsics(2,4) = y;
            extrinsics(3,4) = z;

            saveToYamlFile (fileID, calibrationParameters{iCams(iCam)}.FocalLength, ...
                calibrationParameters{iCams(iCam)}.PrincipalPoint, ...
                calibrationParameters{iCams(iCam)}.RadialDistortion, ...
                calibrationParameters{iCams(iCam)}.TangentialDistortion, ...
                extrinsics, 'rgb');

            if (iCams(iCam) == 1)
                fprintf(staticTransformID, '%s: ', 'ee_chameleon');
                fprintf(staticTransformID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f\n', ...
                x, y, z, qx, qy, qz, qw);
                saveToYamlFile (fileID, [0, 0], [0, 0], [0, 0, 0], [0, 0], eye(4), 'depth');
                saveToYamlFile (fileID, [0, 0], [0, 0], [0, 0, 0], [0, 0], eye(4), 'ir1');
                saveToYamlFile (fileID, [0, 0], [0, 0], [0, 0, 0], [0, 0], eye(4), 'ir2');
                fprintf(fileID, '%s %d\n', 'rgb_width:', 2048);
                fprintf(fileID, '%s %d\n', 'rgb_height:', 1536);
                fprintf(fileID, '%s %d\n', 'depth_width:', 0);
                fprintf(fileID, '%s %d\n', 'depth_height:', 0);
            end
            if (iCams(iCam) == 2)
                fprintf(staticTransformID, '%s: ', 'ee_primesense');
                fprintf(staticTransformID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f\n', ...
                x, y, z, qx, qy, qz, qw);
            end
            if (iCams(iCam) == 4)
                fprintf(staticTransformID, '%s: ', 'ee_d415');
                fprintf(staticTransformID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f\n', ...
                x, y, z, qx, qy, qz, qw);
            end
            if (iCams(iCam) == 7)
                fprintf(staticTransformID, '%s: ', 'ee_d435');
                fprintf(staticTransformID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f\n', ...
                x, y, z, qx, qy, qz, qw);
            end
        end
        if (iCam == 2)
            if (currentSet == 2)
                rotation = extrinsicsCalibrationParameters{1}.RotationOfCamera2;
                translation = extrinsicsCalibrationParameters{1}.TranslationOfCamera2;
            end
            if (currentSet == 3)
                rotation = extrinsicsCalibrationParameters{3}.RotationOfCamera2;
                translation = extrinsicsCalibrationParameters{3}.TranslationOfCamera2;
            end
            if (currentSet == 4)
                rotation = extrinsicsCalibrationParameters{5}.RotationOfCamera2;
                translation = extrinsicsCalibrationParameters{5}.TranslationOfCamera2;
            end

            extrinsics = eye(4);
            extrinsics(1:3, 1:3) = rotation';
            extrinsics(1,4) = translation(1)./1000;
            extrinsics(2,4) = translation(2)./1000;
            extrinsics(3,4) = translation(3)./1000;

            saveToYamlFile (fileID, calibrationParameters{iCams(iCam)}.FocalLength, ...
                calibrationParameters{iCams(iCam)}.PrincipalPoint, ...
                [0, 0, 0], [0, 0], ...
                extrinsics, 'depth');
            
            saveToYamlFile (fileID, calibrationParameters{iCams(iCam)}.FocalLength, ...
                calibrationParameters{iCams(iCam)}.PrincipalPoint, ...
                calibrationParameters{iCams(iCam)}.RadialDistortion, ...
                calibrationParameters{iCams(iCam)}.TangentialDistortion, ...
                extrinsics, 'ir1');

            if (iCams(iCam) == 3)
                quaternion = rotm2quat(rotation');
                fprintf(staticTransformID, '%s: ', 'primesense_ir_to_rgb');
                fprintf(staticTransformID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f\n', ...
                extrinsics(1,4), extrinsics(2,4), extrinsics(3,4), ...
                quaternion(2), quaternion(3), quaternion(4), quaternion(1));
                saveToYamlFile (fileID, [0, 0], [0, 0], [0, 0, 0], [0, 0], eye(4), 'ir2');
                fprintf(fileID, '%s %d\n', 'rgb_width:', 640);
                fprintf(fileID, '%s %d\n', 'rgb_height:', 480);
                fprintf(fileID, '%s %d\n', 'depth_width:', 640);
                fprintf(fileID, '%s %d\n', 'depth_height:', 480);
            end
            if (iCams(iCam) == 5)
                quaternion = rotm2quat(rotation');
                fprintf(staticTransformID, '%s: ', 'd415_ir1_to_rgb');
                fprintf(staticTransformID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f\n', ...
                extrinsics(1,4), extrinsics(2,4), extrinsics(3,4), ...
                quaternion(2), quaternion(3), quaternion(4), quaternion(1));
            end
            if (iCams(iCam) == 8)
                quaternion = rotm2quat(rotation');
                fprintf(staticTransformID, '%s: ', 'd435_ir1_to_rgb');
                fprintf(staticTransformID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f\n', ...
                extrinsics(1,4), extrinsics(2,4), extrinsics(3,4), ...
                quaternion(2), quaternion(3), quaternion(4), quaternion(1));
            end
        end
        if (iCam == 3)
            if (currentSet == 3)
                rotation = extrinsicsCalibrationParameters{2}.RotationOfCamera2;
                translation = extrinsicsCalibrationParameters{2}.TranslationOfCamera2;
            end
            if (currentSet == 4)
                rotation = extrinsicsCalibrationParameters{4}.RotationOfCamera2;
                translation = extrinsicsCalibrationParameters{4}.TranslationOfCamera2;
            end

            extrinsics = eye(4);
            extrinsics(1:3, 1:3) = rotation';
            extrinsics(1,4) = translation(1)./1000;
            extrinsics(2,4) = translation(2)./1000;
            extrinsics(3,4) = translation(3)./1000;

            saveToYamlFile (fileID, calibrationParameters{iCams(iCam)}.FocalLength, ...
                calibrationParameters{iCams(iCam)}.PrincipalPoint, ...
                calibrationParameters{iCams(iCam)}.RadialDistortion, ...
                calibrationParameters{iCams(iCam)}.TangentialDistortion, ...
                extrinsics, 'ir2');

            if (iCams(iCam) == 6)
                quaternion = rotm2quat(rotation');
                fprintf(staticTransformID, '%s: ', 'd415_ir2_to_ir1');
                fprintf(staticTransformID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f\n', ...
                extrinsics(1,4), extrinsics(2,4), extrinsics(3,4), ...
                quaternion(2), quaternion(3), quaternion(4), quaternion(1));
            end
            if (iCams(iCam) == 9)
                quaternion = rotm2quat(rotation');
                fprintf(staticTransformID, '%s: ', 'd435_ir2_to_ir1');
                fprintf(staticTransformID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f\n', ...
                extrinsics(1,4), extrinsics(2,4), extrinsics(3,4), ...
                quaternion(2), quaternion(3), quaternion(4), quaternion(1));
            end
           
            fprintf(fileID, '%s %d\n', 'rgb_width:', 1920);
            fprintf(fileID, '%s %d\n', 'rgb_height:', 1080);
            fprintf(fileID, '%s %d\n', 'depth_width:', 1280);
            fprintf(fileID, '%s %d\n', 'depth_height:', 720);
        end
    end


    fprintf(fileID, '%s %s\n', 'sensor_name:', ['"' cameraSets{currentSet} '"']);
    fprintf(fileID, '%s %s\n', 'serial_number:', ['"' 'n/a' '"']);
    fprintf(fileID, '%s %s\n', 'robot_name:', ['"' 'ur10' '"']);
    if strcmp(cameraSets{currentSet},'primesense')
        fprintf(fileID, '%s %f\n', 'z_scaling:', 1.0325);
    else
        fprintf(fileID, '%s %f\n', 'z_scaling:', 1.0);
    end
    if strcmp(cameraSets{currentSet}, 'primesense')
        fprintf(fileID, '%s %f\n', 'depth_scale_mm:', 1.0);
    elseif strcmp(cameraSets{currentSet}, 'chameleon')
        fprintf(fileID, '%s %f\n', 'depth_scale_mm:', 0.0);
    else
        fprintf(fileID, '%s %f\n', 'depth_scale_mm:', 0.1);
    end
    fclose(fileID);
end

for iCam = 6:8
    rotation = extrinsicsCalibrationParameters{iCam}.RotationOfCamera2;
    translation = extrinsicsCalibrationParameters{iCam}.TranslationOfCamera2;
    quaternion = rotm2quat(rotation');
    if (iCam == 6)
        fprintf(staticTransformID, '%s: ', 'chameleon_to_primesense_rgb');
    end
    if (iCam == 7)
        fprintf(staticTransformID, '%s: ', 'd415_rgb_to_primesense_rgb');
    end
    if (iCam == 8)
        fprintf(staticTransformID, '%s: ', 'd435_rgb_to_primesense_rgb');
    end
    fprintf(staticTransformID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f\n', ...
    translation(1)./1000, translation(2)./1000, translation(3)./1000, ...
    quaternion(2), quaternion(3), quaternion(4), quaternion(1));
end
fclose(staticTransformID);
