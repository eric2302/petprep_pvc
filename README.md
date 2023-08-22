# PETprep: Partial Volume Correction module
This script uses [PETPVC](https://github.com/UCL/PETPVC) on Docker to apply partial volume correction to PET images.
It additionally formats anatomical segmentation files to accomodate the PETPVC format.
The only requirement is to have Docker and Python (>=3.7.13) installed. In order to
use this, clone the repo next to the folder of your BIDS-compatible data by running

    git clone git@github.com:eric2302/petprep_pvc.git

## Requirements
Make sure you have all the needed packages by running
    
    pip install -r requirements.txt

## Use
In order to use this script, call on the terminal
    
    python run_pvc.py --bids_dir BIDS_DIR --participant_label LABEL --pvc_method {GTM,LABBE,RL,VC,STC,MTC,IY,MG} --fwhm FWHM

to run the complete workflow. All the options inlcuded in this current version are:

    --bids_dir BIDS_DIR   The directory with the input dataset formatted
                          according to the BIDS standard.
                          
    --output_dir OUTPUT_DIR
                          The directory where the output files should be stored.
                          If you are running group level analysis this folder
                          should be prepopulated with the results of the
                          participant level analysis.
                          
    --analysis_level {participant,group}
                          Level of the analysis that will be performed. Multiple
                          participant level analyses can be run independently
                          (in parallel) using the same output_dir.
                          
    --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                          The label(s) of the participant(s) that should be
                          analyzed. The label corresponds to
                          sub-<participant_label> from the BIDS spec (so it does
                          not include "sub-"). If this parameter is not provided
                          all subjects should be analyzed. Multiple participants
                          can be specified with a space separated list.
                          
    --pvc_method {GTM,LABBE,RL,VC,STC,MTC,IY,MG}
                          Partial volume correction method.
                          
    --fwhm FWHM           Full width at half maximum in mm for point spread
                          function.
                          
    --version VERSION     Corresponds to the user that hosts the PETPVC
                          container on Docker Hub.
                        
    --skip_anat_prep      Whether or not to prepare anatomical data for PETPVC.
