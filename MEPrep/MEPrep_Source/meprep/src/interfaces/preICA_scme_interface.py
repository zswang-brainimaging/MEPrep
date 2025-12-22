"""
Multi-echo Interface to implement optimally combining, 
denosing with ME-ICA and both functons

Made by Dr. Zhishun Wang in 2021, 2022

This is the interface to Run spatially concatenated ME ICA (scme-ICA)

 melodic -i e1.nii.gz,e2.nii.gz,e3.nii.gz,e4.nii.gz -o tica_out_command --tr=2.47 --mask=mask1.nii.gz -a tica --Oall --Ostats --nobet --mmthresh=0.5 --report
 
 Spatially Concatenated Multi-Echoes with Memory Efficiency (preICA_scmeme): in denoising step, GLM is applied to single-echo rather than merged echoes
 Dr. Wang in Aug 2025

"""

import os
import subprocess
from os.path import join as fullfile

import logging

import nibabel as nib
from nipype.interfaces.base import OutputMultiPath

import numpy.matlib 
import numpy as np

from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File,
    SimpleInterface
)


import shutil

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')

LOGGER = logging.getLogger('nipype.interface')

Remove_tmean = True
Use_global_mean = False

class run_scmeICAInputSpec(BaseInterfaceInputSpec):
    
     out_dir = traits.List(File(exists=False),
                            ['.'],
                            argstr='-o %s',
                            position=1,
                            usedefault=True,
                            minlen=1,
                            desc='Output dir of the preICA results')
     
     in_file = traits.List(File(exists=True),
                           argstr='-i %s',
                           position=2,
                           mandatory=True,
                           minlen=3,
                           desc='multi-echo BOLD EPIs')
     
     mask_file = traits.File(exists=True,
                           argstr='--m %s',
                           position=3,
                           mandatory=True,
                           minlen=1,
                           desc='EPI mask file')
     
     motion_file = traits.File(exists=True,
                            argstr='-mc %s',
                            position=4,
                            mandatory=True,
                            minlen=1,
                            desc='Motion parametric file of multi-echo BOLD EPIs')
     
     bold2mni_matrix_file = traits.File(exists=True,
                            argstr='-a %s',
                            position=5,
                            mandatory=True,
                            minlen=1,
                            desc='Matrix file to register t-mean echo1-BOLD to MNI152 space')
     
     TR = traits.Float(
                            argstr='-tr %.10f',
                            position=6,
                            mandatory=True,
                            desc='TR (second) of multi-echo BOLD EPIs')
     dim = traits.Int(
                            0,
                            argstr='-dim %d',
                            position=7,
                            usedefault=True,
                            desc='Dimensionality reduction into #num dimensions when running MELODIC (default: automatic estimation; i.e. dim=0 ')
     
     denoising_type = traits.Enum('nonaggr', 'aggr','both', 'no',
                           argstr='-den %s',
                           position=8,
                           usedefault=True,
                           desc=('Type of denoising strategy: ',
                                 'no: only classification, no denoising;', 
                                 'nonaggr: non-aggresssive denoising (default)',
                                 'aggr: aggressive denoising; ',
                                 'both: both aggressive and non-aggressive denoising (seprately)'
                                 )
                           )
     
     plot_component = traits.Enum('np', 'noplots',
                           argstr='-%s',
                           position=9,
                           usedefault=True,
                           desc=('Plot (np) or not (noplots) component classification overview ',
                                 'similar to plot in the main AROMA paper')
                                  )

class run_scmeICAOutputSpec(TraitedSpec):
   denoised_func_files = OutputMultiPath(File(exists=True), desc='denoised EPI file, i.e.,denoised_func_data_nonaggr.nii.gz')
  
           
      
class run_scmeICA(SimpleInterface):
    
    input_spec = run_scmeICAInputSpec
    output_spec = run_scmeICAOutputSpec
    
    def _run_interface(self, runtime):
        
        in_files = self.inputs.in_file
        
        mask_file = self.inputs.mask_file
        
        TR = self.inputs.TR
        
        motion_file = self.inputs.motion_file
        
        bold2mni_matrix_file = self.inputs.bold2mni_matrix_file
        
        echo_N = len(in_files)
        
        out_dir = os.getcwd()
        
        out_dir_parent, out_dir_folderName = os.path.split(out_dir)
        out_dir_backup = fullfile(out_dir_parent, out_dir_folderName + '_backup')
        
        ## ZSW Added in Sept 2025, optimize pOptCom and pMEICA to save time without running preICA twice in case "Both" option is not used
        prior_denoised_echo_list = []
        prior_aroma_outDir =  fullfile(out_dir_backup, 'aromaOut');
        prior_denoised_echo_N = 0
        for tn in range(echo_N):
            echo_Out_str = "Out_e" + str(tn+1)
            echo_tarDir = fullfile(prior_aroma_outDir, echo_Out_str)
            denoised_echo_file = fullfile(echo_tarDir, 'denoised_func_data_nonaggr.nii.gz')
            if os.path.isdir(echo_tarDir) and os.path.exists(denoised_echo_file):
                prior_denoised_echo_N = prior_denoised_echo_N +1
                prior_denoised_echo_list.append(denoised_echo_file)
                LOGGER.log(25, f'Prior denoised echo file {denoised_echo_file} exists!')
            
        if prior_denoised_echo_N==echo_N:
            self._results['denoised_func_files'] = prior_denoised_echo_list
            LOGGER.log(25, f'We found all {echo_N} prior denoised echo files, skipping preICA step !')
            return runtime
        ## ZSW Added in Sept 2025, optimize pOptCom and pMEICA to save time without running preICA twice in case "Both" option is not used          
        
        scmeICA_outDir =  fullfile(out_dir, 'scmeICA_Out');
        if not os.path.exists(scmeICA_outDir):
             os.makedirs(scmeICA_outDir)
        
       ## melDir = os.path.join(scmeICA_outDir, 'melodic.ica')    
        
        """
        mask_img = nib.load(mask_file)
        mask = mask_img.get_data()  #np.array(mask_img.dataobj)
        mask = mask.reshape(-1)
        mask_ind = mask>0
        mask_len = mask_ind.sum()
        img0 = nib.load(in_files[0])
        sx,sy,sz,sT = img0.shape
        scmY_data = np.zeros((echo_N*mask_len, sT), dtype=img0.get_data_dtype())  ## spatially concatenated & masked data
        
        """  
        echo1_img = nib.load(in_files[0])
        dimx,dimy,dimz,dimT = echo1_img.shape
        
        echo_dmean_file_list = []
        echo_tmean_file_list = []   
        echo_global_mean_list = []
        echo_mask_file_list = []
        
        for tn in range(echo_N):
            echo_tmean_str = "e" + str(tn+1) + "_tmean.nii.gz"
            echo_tmean_file = fullfile(out_dir, echo_tmean_str)
            echo_file = in_files[tn]
            com_str = "fslmaths " + echo_file + " -Tmean " + echo_tmean_file
            os.system(com_str)
            
            ## get global mean 
            ## gM1=$(fslstats e1_tmean.nii.gz -M)
            com_str = "fslstats " + echo_tmean_file + " -M"
            proc = subprocess.Popen([com_str], stdout=subprocess.PIPE, shell=True)
            (gMean_str , err) = proc.communicate()
            gMean = float(gMean_str)  
            gMean_name = "gMean_e" + str(tn+1)
            exec("%s = %f" % (gMean_name, gMean))
            
            echo_dmean_str = "e" + str(tn+1) + "_dmean.nii.gz"
            echo_dmean_file = fullfile(out_dir, echo_dmean_str)
            
            if not Use_global_mean:
                com_str = "fslmaths " + echo_file + " -sub " + echo_tmean_file + " " + echo_dmean_file
            else:
                com_str = "fslmaths " + echo_file + " -sub " + str(gMean) + " " + echo_dmean_file   
                
            os.system(com_str)
            echo_dmean_file_list.append(echo_dmean_file)
            echo_tmean_file_list.append(echo_tmean_file)
            echo_global_mean_list.append(gMean)
            echo_mask_file_list.append(mask_file)
            
            ## read echo image, mask it and merge the masked data spatially 
            #img = nib.load(in_files[tn])
            #X_img = img.get_data()
            #X_img = X_img.reshape(sx*sy*sz, sT)
            #Y_img = X_img[mask_ind,:]
            #scmY_data[tn*mask_len:(tn+1)*mask_len,:] = Y_img
        
        ## merge multiple echos into one nii file along time
        
       # fslmerge -t dm_e1234 dm_e1 dm_e2 dm_e3 dm_e4
        
        echos_merged_file = fullfile(scmeICA_outDir, "echos_merged_file.nii.gz")
        echos_merged_mask_file = fullfile(scmeICA_outDir, "echos_merged_mask_file.nii.gz")
       
       # scmY_data_4D = scmY_data.reshape(mask_len*echo_N,1,1,sT)
       # merged_img = nib.Nifti1Image(scmY_data_4D, img.affine, nib.Nifti1Header())
       # nib.save(merged_img, echos_merged_file)
        
        if not Remove_tmean:
            com_str = "fslmerge -z " + echos_merged_file + " " + " ".join(in_files)
        else:
            com_str = "fslmerge -z " + echos_merged_file + " " + " ".join(echo_dmean_file_list)
        os.system(com_str)
        
        com_str = "fslmerge -z " + echos_merged_mask_file + " " + " ".join(echo_mask_file_list)
        os.system(com_str)
        
        #melodic -i e1.nii.gz,e2.nii.gz,e3.nii.gz,e4.nii.gz -o tica_out_command --tr=2.47 --mask=mask1.nii.gz -a tica --Oall --Ostats --nobet --mmthresh=0.5 --report
        
        ## run melodic : spatially cat ICA
        com_str = "melodic -i " + echos_merged_file + " -o " + scmeICA_outDir + " --tr=" + str(TR) + " --mask=" + echos_merged_mask_file + " --Oall --Ostats --nobet --mmthresh=0.5 --report"
        os.system(com_str) 
        
        create_thresholded_IC(scmeICA_outDir, mask_file, dimx, dimy, dimz)
        
        ## extend motion file for multiple echos 
        single_echo_M = np.loadtxt(motion_file)
        multi_echo_M = single_echo_M   # np.matlib.repmat(single_echo_M, echo_N, 1)
        ## for spatially cat data, no need to extend the motion file
        multi_echo_motion_file = fullfile(scmeICA_outDir, "multi_echo_motion.txt")
        np.savetxt(multi_echo_motion_file, multi_echo_M)
        
        ## now run ICA_AROMA.py
        
        # ICA_AROMA.py  -o out -i merged_echos -mc motion_file  -a bold2nii.mat -m mask -tr 2.47 -den nonaggr -dim  0 -md tica_out_dir
        aroma_outDir =  fullfile(out_dir, 'aromaOut');
        if os.path.exists(aroma_outDir):
             shutil.rmtree(aroma_outDir)
       
         
        com_str = "ICA_AROMA_preICA.py -o " + aroma_outDir + " -i " + echos_merged_file + " -mc " + multi_echo_motion_file
        com_str = com_str  + " -a " + bold2mni_matrix_file + " -m " + echos_merged_mask_file + " -tr " + str(TR)
       ## com_str = com_str + " -md " + scmeICA_outDir + " -den nonaggr -dim  0"  #'no': only classification, no denoising; 'nonaggr': non-aggresssive denoising
        com_str = com_str + " -md " + scmeICA_outDir + " -den no -dim  0"   ## only classification
        os.system(com_str)
        
        classified_motion_ICs_file = fullfile(aroma_outDir, "classified_motion_ICs.txt")
        fp_ic = open(classified_motion_ICs_file, "r")
        classified_motion_ICs_indices = fp_ic.read()
        fp_ic.close()
        
        ICA_melodic_mix_file = fullfile(scmeICA_outDir, 'melodic_mix')
        
        denoised_echo_list = []
        for tn in range(echo_N):
            echo_Out_str = "Out_e" + str(tn+1)
            echo_tarDir = fullfile(aroma_outDir, echo_Out_str)
            if not os.path.exists(echo_tarDir):
                 os.makedirs(echo_tarDir)
            
            echo_denoised_file_before_adding_tmean = fullfile(echo_tarDir, "denoised_func_before_adding_tmean.nii.gz")
            echo_denoised_file = fullfile(echo_tarDir, "denoised_func_data_nonaggr.nii.gz")
           
            
            ## fsl_regfilt --in  <input> --design  <design> --filter <component numbers or filter threshold> --out <out> [options]
            if not Remove_tmean:
                input_echo_file_for_denoising = in_files[tn]
            else:
                input_echo_file_for_denoising = echo_dmean_file_list[tn]
                
            com_str = 'fsl_regfilt --in=' + input_echo_file_for_denoising + ' --design=' + ICA_melodic_mix_file
            com_str = com_str + ' --filter="' + classified_motion_ICs_indices + '"'
            if not Remove_tmean:
                com_str = com_str + ' --out=' + echo_denoised_file
            else:
                com_str = com_str + ' --out=' + echo_denoised_file_before_adding_tmean 
            
            ## Perform denoising 
            LOGGER.log(25, f'Perform denoising with command  {com_str} !')
            os.system(com_str)
            com_file = fullfile(echo_tarDir, 'command.txt')
            with open(com_file, "w") as fp_write:
                 fp_write.write(com_str)  
           
            if Remove_tmean:
               echo_tmean_str = "e" + str(tn+1) + "_tmean.nii.gz"
               echo_tmean_file = fullfile(out_dir, echo_tmean_str)
               if not Use_global_mean:
                  com_str = "fslmaths " + echo_denoised_file_before_adding_tmean + " -add " + echo_tmean_file + " " + echo_denoised_file
               else:
                   gMean = echo_global_mean_list[tn]
                   com_str = "fslmaths " + echo_denoised_file_before_adding_tmean + " -add " + str(gMean) + " " + echo_denoised_file
               os.system(com_str)
              
            denoised_echo_list.append(echo_denoised_file)
        
        
        self._results['denoised_func_files'] = denoised_echo_list
                     
        ## backup the preICA outputs:
        dst_aroma_outDir =  fullfile(out_dir_backup, 'aromaOut');    
        src_aroma_outDir =  fullfile(out_dir, 'aromaOut');
        if os.path.exists(dst_aroma_outDir):
             shutil.rmtree(dst_aroma_outDir)
        shutil.copytree(src_aroma_outDir, dst_aroma_outDir)     
        ### END of backup the preICA outputs
        
        return runtime
        
def create_thresholded_IC(outDir, echo1_mask, dimx, dimy, dimz):
    fslDir = os.path.join(os.environ["FSLDIR"], 'bin', '')
    
    melDir = outDir
    melIC = os.path.join(melDir, 'melodic_IC.nii.gz')
    melICthr = os.path.join(outDir, 'melodic_IC_thr.nii.gz')
   
    # Get number of components
    cmd = ' '.join([os.path.join(fslDir, 'fslinfo'),
                    melIC,
                    '| grep dim4 | head -n1 | awk \'{print $2}\''])
    nrICs = int(float(subprocess.getoutput(cmd)))

    # Merge mixture modeled thresholded spatial maps. Note! In case that mixture modeling did not converge, the file will contain two spatial maps. The latter being the results from a simple null hypothesis test. In that case, this map will have to be used (first one will be empty).
    for i in range(1, nrICs + 1):
        # Define thresholded zstat-map file
        zTemp = os.path.join(melDir, 'stats', 'thresh_zstat' + str(i) + '.nii.gz')
        cmd = ' '.join([os.path.join(fslDir, 'fslinfo'),
                        zTemp,
                        '| grep dim4 | head -n1 | awk \'{print $2}\''])
        lenIC = int(float(subprocess.getoutput(cmd)))

        # Define zeropad for this IC-number and new zstat file
        cmd = ' '.join([os.path.join(fslDir, 'zeropad'),
                        str(i),
                        '4'])
        ICnum = subprocess.getoutput(cmd)
        zstat = os.path.join(outDir, 'thr_zstat' + ICnum)

        # Extract last spatial map within the thresh_zstat file
        os.system(' '.join([os.path.join(fslDir, 'fslroi'),
                            zTemp,      # input
                            zstat,      # output
                            '0',
                            str(dimx),
                            '0',
                            str(dimy),
                            '0',
                            str(dimz),
                            str(lenIC - 1),   # first frame
                            '1']))      # number of frames

    # Merge and subsequently remove all mixture modeled Z-maps within the output directory
    os.system(' '.join([os.path.join(fslDir, 'fslmerge'),
                        '-t',                       # concatenate in time
                        melICthr,                   # output
                        os.path.join(outDir, 'thr_zstat????.nii.gz')]))  # inputs

    os.system('rm ' + os.path.join(outDir, 'thr_zstat????.nii.gz'))

    # Apply the mask to the merged file (in case a melodic-directory was predefined and run with a different mask)
    os.system(' '.join([os.path.join(fslDir, 'fslmaths'),
                        melICthr,
                        '-mas ' + echo1_mask,
                        melICthr]))      
        
        