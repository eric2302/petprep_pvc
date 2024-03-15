# %%
# %%
import argparse
import os
import warnings
import numpy as np
import subprocess
import nilearn.image as ni
import nibabel as nib
from bids import BIDSLayout

# %%
def main(args):


    if os.path.exists(args.bids_dir):
        # not validated until derivatives structure is definded in BEP23
        layout = BIDSLayout(args.bids_dir, validate=False)
    else:
        raise Exception('BIDS directory does not exist')
    
    # get all PET files if no label is given
    if args.participant_label is None:
        args.participant_label = layout.get(suffix='pet', space='T1w', target='subject', return_type='id')

    # create derivatives directories
    if args.output_dir is None:
        output_dir = os.path.join(args.bids_dir,'derivatives','petprep_pvc')
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # index all sessions and participants
    sessions = layout.get_sessions()
    participants = args.participant_label
    
    # create prefix for filenames
    if not sessions:
        file_prefix = [f'sub-{sub_id}' 
                       for sub_id in participants]
    else:
        file_prefix = [f'sub-{sub_id}_ses-{sess_id}' 
                       for sub_id, sess_id in zip(participants, sessions)]

    # whether to prepare anatomical data for PETPVC
    # whether to prepare anatomical data for PETPVC
    if not args.skip_anat_prep:
        # create 4D tissue segmentation (fourth dimension should add up to 1)
        # create 4D tissue segmentation (fourth dimension should add up to 1)
        for fp in file_prefix:
            if not sessions:
                subj_dir = os.path.join(args.bids_dir, fp)
                subj_out_dir = os.path.join(output_dir, fp)
                subj_out_dir = os.path.join(output_dir, fp)
            else:
                sub, ses = fp.split('_')
                subj_dir = os.path.join(args.bids_dir, sub, ses)
                subj_out_dir = os.path.join(output_dir, sub, ses)

            os.makedirs(subj_out_dir, exist_ok=True)
            
            # find directory with anatomical data
            anat_dir = find_anat_dir(args.bids_dir, subj_dir)
            gm_prob = os.path.join(anat_dir, f'{fp}_label-GM_probseg.nii.gz')
            wm_prob = os.path.join(anat_dir, f'{fp}_label-WM_probseg.nii.gz')
            csf_prob = os.path.join(anat_dir, f'{fp}_label-CSF_probseg.nii.gz')
            subj_out_dir = os.path.join(output_dir, sub, ses)

            os.makedirs(subj_out_dir, exist_ok=True)
            
            # find directory with anatomical data
            anat_dir = find_anat_dir(args.bids_dir, subj_dir)
            gm_prob = os.path.join(anat_dir, f'{fp}_label-GM_probseg.nii.gz')
            wm_prob = os.path.join(anat_dir, f'{fp}_label-WM_probseg.nii.gz')
            csf_prob = os.path.join(anat_dir, f'{fp}_label-CSF_probseg.nii.gz')
            
            path_exists = [os.path.exists(gm_prob),
                           os.path.exists(wm_prob),
                           os.path.exists(csf_prob)]
            
            if not np.all(path_exists):
                raise Exception('Check that {GM,WM,CSF}_probseg exists in anat directory: ', anat_dir)
            
            prepare_anat(fp, gm_prob, wm_prob, csf_prob, subj_out_dir)
            
            
    # run PETPVC for every session and participant included in the analysis
    for fp in file_prefix:
        if not sessions:
            subj_dir = os.path.join(args.bids_dir, fp)
            subj_out_dir = os.path.join(output_dir, fp)
            pet_dir = os.path.join(args.bids_dir, 'derivatives', 'petprep_extract_tacs', fp)
            subj_out_dir = os.path.join(output_dir, fp)
            pet_dir = os.path.join(args.bids_dir, 'derivatives', 'petprep_extract_tacs', fp)
        else:
            sub, ses = fp.split('_')
            subj_dir = os.path.join(args.bids_dir, sub, ses)
            subj_out_dir = os.path.join(output_dir, sub, ses)
            pet_dir = os.path.join(args.bids_dir, 'derivatives', 'petprep_extract_tacs', sub, ses)
        
        anat_dir = find_anat_dir(args.bids_dir, subj_dir)
        t1 = os.path.join(anat_dir, f'{fp}_desc-preproc_T1w.nii.gz')
        brain_mask = os.path.join(anat_dir, f'{fp}_desc-brain_mask.nii.gz')
        
        subj_out_dir = os.path.abspath(subj_out_dir)
        pet_dir = os.path.abspath(pet_dir)
        pet_fn = f'{fp}_space-T1w_desc-twa_pet.nii.gz'
        
        resample_pet_to_native(pet_dir, pet_fn, subj_out_dir, t1)
        mask_pet(pet_fn, brain_mask, subj_out_dir)
        
        seg_fn = f'{fp}_desc-4Danatseg.nii.gz'
        if not os.path.exists(os.path.join(subj_out_dir, seg_fn)):
            raise Exception(f"No tissue segmentation in output folder for {fp}. "
                            f"Make sure it is named '{fp}_desc-4Danatseg.nii.gz'")                       

        method = args.pvc_method.lower()
        pet_pvc_fn = f'{fp}_space-T1w_pvc-{method}_desc-preproc_pet.nii.gz'
        cmd = ("petpvc "
               f"-i {subj_out_dir}/{pet_fn} "
               f"-m {subj_out_dir}/{seg_fn} "
               f"-o {subj_out_dir}/{pet_pvc_fn} "
               f"-p {args.pvc_method} "
               f"-x {args.fwhm} -y {args.fwhm} -z {args.fwhm}")
        
        print("Running PETPVC")
        result = subprocess.run(cmd, shell=True, text=True, 
                                stderr=subprocess.STDOUT)

# HELPER FUNCTIONS

def find_anat_dir(bids_dir, subj_dir):
    subj_path = os.path.relpath(subj_dir, bids_dir)

    # check if smriprep directory exists in the derivatives folder
    smriprep_dir = os.path.join(bids_dir, 'derivatives', 'smriprep', subj_path)

    if os.path.exists(smriprep_dir):
        # return anat directory in the smriprep folder
        anat_dir = os.path.join(bids_dir, 'derivatives', 'smriprep', subj_path, 'anat')
        return anat_dir
    else:
        # return anat directory in the subject folder
        anat_dir = os.path.join(subj_dir, 'anat')
        return anat_dir

# HELPER FUNCTIONS

def find_anat_dir(bids_dir, subj_dir):
    subj_path = os.path.relpath(subj_dir, bids_dir)

    # check if smriprep directory exists in the derivatives folder
    smriprep_dir = os.path.join(bids_dir, 'derivatives', 'smriprep', subj_path)

    if os.path.exists(smriprep_dir):
        # return anat directory in the smriprep folder
        anat_dir = os.path.join(bids_dir, 'derivatives', 'smriprep', subj_path, 'anat')
        return anat_dir
    else:
        # return anat directory in the subject folder
        anat_dir = os.path.join(subj_dir, 'anat')
        return anat_dir

def prepare_anat(sub, gm_prob, wm_prob, csf_prob, output_dir):
    print(f"Preparing segmentation for {sub}")
    # dividing by zero - turning off warnings temporarily
    warnings.filterwarnings("ignore", category=RuntimeWarning)  
                
    norm_gm  =  ni.math_img("img1 / (img1 + img2 + img3)", 
                img1=gm_prob,
                img2=wm_prob,
                img3=csf_prob)
    data = np.nan_to_num(norm_gm.get_fdata())
    norm_gm = ni.new_img_like(gm_prob, data)

    norm_wm  =  ni.math_img("img2 / (img1 + img2 + img3)", 
                img1=gm_prob,
                img2=wm_prob,
                img3=csf_prob)
    data = np.nan_to_num(norm_wm.get_fdata())
    norm_wm = ni.new_img_like(wm_prob, data)

    norm_csf  = ni.math_img("img3 / (img1 + img2 + img3)", 
                img1=gm_prob,
                img2=wm_prob,
                img3=csf_prob)
    data = np.nan_to_num(norm_csf.get_fdata())
    norm_csf = ni.new_img_like(csf_prob, data)

    seg_fn = os.path.join(output_dir, f'{sub}_desc-4Danatseg.nii.gz')
    seg_fn = os.path.join(output_dir, f'{sub}_desc-4Danatseg.nii.gz')
    seg = ni.concat_imgs([norm_gm, norm_wm, norm_csf])
    seg.to_filename(seg_fn)

    # reset warnings
    warnings.resetwarnings()
    
    return None

def resample_pet_to_native(pet_dir, pet_fn, subj_out_dir, t1):
    pet = os.path.join(pet_dir, pet_fn)
    
    pet_rs = ni.resample_to_img(pet, t1, interpolation='continuous', clip=True)
    pet_rs = ni.new_img_like(t1, pet_rs.get_fdata())
    pet_rs.to_filename(os.path.join(subj_out_dir, pet_fn))
    
    return None

def mask_pet(pet_fn, brain_mask, subj_out_dir):
    pet_fn = os.path.join(subj_out_dir, pet_fn)
    mask = ni.math_img("img > 0", 
                       img=brain_mask)
    
    pet_masked = ni.math_img("img1 * img2", 
                             img1=pet_fn, 
                             img2=mask)
    pet_masked.to_filename(pet_fn)
    
    return None

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='BIDS App for PETPrep partial volume correction.')
    parser.add_argument('--bids_dir', required=True,  help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
    parser.add_argument('--output_dir', required=False, help='The directory where the output files '
                    'should be stored. If you are running group level analysis '
                    'this folder should be prepopulated with the results of the '
                    'participant level analysis.')
    parser.add_argument('--analysis_level', default='participant', help='Level of the analysis that will be performed. '
                    'Multiple participant level analyses can be run independently '
                    '(in parallel) using the same output_dir.',
                    choices=['participant', 'group'])
    parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                   'corresponds to sub-<participant_label> from the BIDS spec '
                   '(so it does not include "sub-"). If this parameter is not '
                   'provided all subjects should be analyzed. Multiple '
                   'participants can be specified with a space separated list.',
                   nargs="+", default=None)
    parser.add_argument('--pvc_method', help='Partial volume correction method.',
                   required=True, default='IY',
                   choices = ['GTM', 'LABBE','RL', 'VC', 'STC', 'MTC', 'IY', 'MG'])
    parser.add_argument('--fwhm', help='Full width at half maximum in mm for point spread function.',
                   required=True, default='6')
    parser.add_argument('--version', help='Corresponds to the user that hosts the PETPVC container on Docker Hub.',
                    default='aramislab')
    parser.add_argument('--skip_anat_prep', help='Whether or not to prepare anatomical data for PETPVC.',
                   action='store_true', default=False)
    args = parser.parse_args() 
    
    main(args)