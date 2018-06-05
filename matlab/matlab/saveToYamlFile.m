function saveToYamlFile (fileID, focalLength, principalPoint, ...
    radialDistortion, tangentialDistortion, extrinsics, name)


    fprintf(fileID, '%s%s: !!opencv-matrix\n', name, '_intrinsics');
    fprintf(fileID, '   %s: %d\n', 'rows', 3);
    fprintf(fileID, '   %s: %d\n', 'cols', 3);
    fprintf(fileID, '   %s: %s\n', 'dt', 'd');
    fprintf(fileID, '   %s: [', 'data');
    fprintf(fileID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f]\n', ...
        focalLength(1), 0.0, principalPoint(1), 0.0, focalLength(2), ...
        principalPoint(2), 0.0, 0.0, 1.0);
    
    fprintf(fileID, '%s%s: !!opencv-matrix\n', name, '_distortion_coeffs');
    fprintf(fileID, '   %s: %d\n', 'rows', 1);
    fprintf(fileID, '   %s: %d\n', 'cols', 5);
    fprintf(fileID, '   %s: %s\n', 'dt', 'd');
    fprintf(fileID, '   %s: [', 'data');
    fprintf(fileID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f]\n', ...
        radialDistortion(1), radialDistortion(2), tangentialDistortion(1), ...
        tangentialDistortion(2), radialDistortion(3));
    
    if strcmp(name,'rgb')
        variableName = 'hand_eye_transform';
    else
        variableName = [name '_extrinsics'];
    end
    
    fprintf(fileID, '%s: !!opencv-matrix\n', variableName);
    fprintf(fileID, '   %s: %d\n', 'rows', 4);
    fprintf(fileID, '   %s: %d\n', 'cols', 4);
    fprintf(fileID, '   %s: %s\n', 'dt', 'd');
    fprintf(fileID, '   %s: [', 'data');
    fprintf(fileID, '%4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f, %4.12f]\n', ...
        extrinsics(1,1), extrinsics(1,2), extrinsics(1,3), extrinsics(1,4), ...
        extrinsics(2,1), extrinsics(2,2), extrinsics(2,3), extrinsics(2,4), ...
        extrinsics(3,1), extrinsics(3,2), extrinsics(3,3), extrinsics(3,4), ...
        extrinsics(4,1), extrinsics(4,2), extrinsics(4,3), extrinsics(4,4));
    

end