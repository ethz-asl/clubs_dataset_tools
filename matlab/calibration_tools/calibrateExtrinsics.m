function [calibrationParameters, estimationErrors, extrinsicsImagesUsed] = calibrateExtrinsics ...
    (imagePointsArrayCam1, imagePointsArrayCam2, ...
    imagesUsedCam1, imagesUsedCam2, worldPoints, displayReprojectionError, ...
    displayExtrinsics, iCams)


    usedImages = [imagesUsedCam1, imagesUsedCam2];
    extrinsicsImagesUsed = min(usedImages,[],2);
    notUsedIndices = find(extrinsicsImagesUsed==0);

    indicesToRemove1 = [];
    indicesToRemove2 = [];
    for idx = 1 : length(notUsedIndices)
        if (imagesUsedCam1(notUsedIndices(idx)) ~= 0)
            indicesToRemove1 = [indicesToRemove1; sum(imagesUsedCam1(1:notUsedIndices(idx)))];
        end
        if (imagesUsedCam2(notUsedIndices(idx)) ~= 0)
            indicesToRemove2 = [indicesToRemove2; sum(imagesUsedCam2(1:notUsedIndices(idx)))];
        end
    end

    imagePoints1 = imagePointsArrayCam1;
    imagePoints2 = imagePointsArrayCam2;

    imagePoints1(:,:,indicesToRemove1) = [];
    imagePoints2(:,:,indicesToRemove2) = [];

    imagePointsPair = cat(4, imagePoints1, imagePoints2);
    [calibrationParameters, ~, estimationErrors] = ...
            estimateCameraParameters(imagePointsPair, worldPoints, 'EstimateSkew', ...
            false, 'EstimateTangentialDistortion', true, 'NumRadialDistortionCoefficients', ...
            3, 'WorldUnits', 'mm');
    
    if displayReprojectionError
        figure; 
        showReprojectionErrors(calibrationParameters); 
        title(['Mean Reprojection Errors Cam  ' num2str(iCams(1)) ' and ' num2str(iCams(2))]);
        drawnow;
    end
     
    if displayExtrinsics
        figure; 
        showExtrinsics(calibrationParameters); 
        title(['Extrinsics Cam ' num2str(iCams(1)) ' and ' num2str(iCams(2))]);
        drawnow;
    end 

end