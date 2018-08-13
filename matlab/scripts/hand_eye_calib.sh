#!/bin/sh

if [ "$#" -ne 6 ]; then
  echo "THIS SCRIPT SHOULD ONLY BE USED THROUGH MATLAB WHICH PROVIDES IT WITH THE CORRECT INPUTS" >&2
  exit 1
fi

source $1
source $2

touch time_offset.csv
echo "0.0" > time_offset.csv
python $3/compute_hand_eye_calibration.py --aligned_poses_B_H_csv_file $4 --aligned_poses_E_W_csv_file $5 --time_offset_input_csv_file time_offset.csv --calibration_output_json_file $6
rm time_offset.csv
