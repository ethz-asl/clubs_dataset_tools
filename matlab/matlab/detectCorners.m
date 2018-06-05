function [imagePointsArray, boardSizeArray, imagesUsedArray] = detectCorners (imageLocationArray, displayDetectedCorners, iCam)

    [imagePointsArray, boardSizeArray, imagesUsedArray] = detectCheckerboardPoints(imageLocationArray);
    counter = 1;
    if displayDetectedCorners
        figure;
        title(['Detected corners for camera ' num2str(iCam)]);
        for i = 1:numel(imageLocationArray)
            I = imread(imageLocationArray{i});
            subplotX = ceil(sqrt(numel(imageLocationArray)/0.6));
            subplotY = round(subplotX * 0.6);
            subplot(subplotY, subplotX, i);
            imshow(I);
            hold on;
            parsed_path = strsplit(imageLocationArray{i},'/');
            parsed_path = strsplit(parsed_path{end}, '.');
            title(parsed_path{1})
            if imagesUsedArray(i)
                plot(imagePointsArray(:,1,counter),imagePointsArray(:,2,counter),'ro');
                counter = counter + 1;
            end
            hold off;
        end
    end
    drawnow;
    
end