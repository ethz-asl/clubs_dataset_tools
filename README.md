# CLUBS Dataset Tools
TODO



# Requirements

To use the hand-eye calibration please follow the instructions on the hand_eye_calibration package website:
https://github.com/ethz-asl/hand_eye_calibration.
Note that only hand_eye_calibration package is necessary, all the other packages can be ignored.


To install all the python requirements run:
```
pip install -r requirements.txt
```

# Calibration
In order to run the calibration, all the data (images and poses) needs to be placed in matlab/data folder.
![alt text](https://github.com/ethz-asl/clubs_dataset_tools/blob/feature/cloud_generation/images/data_folder_structure.png)

Furthermore, if you wish to run the hand-eye calibration, hand_eye_calibration package from https://github.com/ethz-asl/hand_eye_calibration needs to be compiled and sourced. To run the matlab script, only hand_eye_calibration package is necessary, hand_eye_calibration_batch_estimation, hand_eye_calibration_experiments and hand_eye_calibration_target_extractor packages can therefore be ignored (by adding CATKIN_IGNORE).

Before starting the calibration, several parameters in the calibration_script.m file need to be set:

- If you are running the calibration for the first time, following command needs to be set to true:
increaseBrightnessOfIRImage = true;
Since IR images coming from the Primesense sensor are very dark, target detector cannot detect any corners required for the calibration. Therefore, brightness of these images is increased. After first execution this parameter should be set back to false otherwise it will increase brightness of the IR images each time calibration_script.m is run.

- You need to specify your shell path, e.g.:
shellPath = '~/.bashrc';

- You need to specify your catkin workspace devel/setup.bash file, e.g.:
handEyeCalibrationWorkspace = '~/catkin_ws/devel/setup.bash';

- You need to specify the path to the hand_eye_calibration bin folder, e.g.:
handEyeCalibrationPath = '~/catkin_ws/src/hand_eye_calibration/hand_eye_calibration/bin';

Once these are set, calibration can be started by executing the calibration_script.m file.
