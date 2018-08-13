function increaseBrightnessIRImages(imageLocationArray, factor)

    for imageIdx = 1:length(imageLocationArray)
        image = imread(imageLocationArray{imageIdx});
        image = image * factor;
        imwrite(image,imageLocationArray{imageIdx});
    end

end
