function [calibrationParameters, imagesToUse, estimationErrors] = calibrateIntrinsics ...
    (imagePointsArray, worldPoints, displayReprojectionError, displayExtrinsics, removeOutliers, iCam)

    [calibrationParameters, imagesToUse, estimationErrors] = ...
        estimateCameraParameters(imagePointsArray, worldPoints, 'EstimateSkew', ...
        false, 'EstimateTangentialDistortion', true, 'NumRadialDistortionCoefficients', ...
        3, 'WorldUnits', 'mm');
    
     if displayReprojectionError
        figure; 
        subplot(1, 2, 1); 
        showReprojectionErrors(calibrationParameters); 
        title(['Mean Reprojection Errors Cam  ' num2str(iCam)]);
        subplot(1, 2, 2); 
        showReprojectionErrors(calibrationParameters, 'ScatterPlot'); 
        title(['Reprojection Errors in Pixels Cam  ' num2str(iCam)]);
        drawnow;
     end
     
     if displayExtrinsics
        figure; 
        showExtrinsics(calibrationParameters,'patternCentric'); 
        title(['Extrinsics Cam ', num2str(iCam)]);
        drawnow;
     end
     
     if removeOutliers
        errors = calibrationParameters.ReprojectionErrors;
        euclideanErrors = mean(sqrt(power(errors(:,1,:),2)+power(errors(:,2,:),2)));
        euclideanErrors = euclideanErrors(:);
        outlierList = isoutlier(euclideanErrors,'mean');
        imagesToUse = imagesToUse - outlierList;
     end

end