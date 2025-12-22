"""
Multi-echo Interface to implement optimally combining, 
denosing with ME-ICA and both functons

Made by Dr. Zhishun Wang in 2021, 2022

This is the interface to Run Tensor ICA 

 melodic -i e1.nii.gz,e2.nii.gz,e3.nii.gz,e4.nii.gz -o tica_out_command --tr=2.47 --mask=mask1.nii.gz -a tica --Oall --Ostats --nobet --mmthresh=0.5 --report

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

Remove_tmean = True        ## remove mean along temporal direction
Use_global_mean = False     ## remove t-mean but only global mean (mean of t-mean)

Apply_AROMA_to_MergedData = False
Apply_AROMA_to_Echo1_Only = True

Use_OuterProduct_for_denosing = False  
### Use outer product of temporal mode @ IC mode @ echo mode to replace fsl_regfilt 

class run_TensorICAInputSpec(BaseInterfaceInputSpec):
    
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

class run_TensorICAOutputSpec(TraitedSpec):
   denoised_func_files = OutputMultiPath(File(exists=True), desc='denoised EPI file, i.e.,denoised_func_data_nonaggr.nii.gz')
  
           
def TensorICA_Denoise_UseOutProduct(echo_tarDir, tICA_outDir, melodic_M_echo, src_echo_file, motion_IC_file):
    """
    
    Y_noise = outer_product((a,b),c)
    Y_clean = Y - Y_noise
    a:  temporal mode
    b:  ICA component mode
    c:  echo  mode
    
    Made by Dr. ZS Wang ZSW Sept, 2023
    
    Parameters
    ----------
    echo_tarDir : folder that stores the denoised image data
    
    tICA_outDir : folder that stores the tensor-ICA result data
        
    melodic_M_echo : combined temporal-mode & echo-mode data 
        
    src_echo_file : single-echo image file to be denoised

    """
    echo_denoised_file = fullfile(echo_tarDir, "denoised_func_data_nonaggr.nii.gz")
    echo_denoised_file_regfilt = fullfile(echo_tarDir, "denoised_func_data_nonaggr_regfilt.nii.gz")
    os.replace(echo_denoised_file, echo_denoised_file_regfilt)
    cf = motion_IC_file
    ## cf=fullfile(echo_tarDir, 'classified_motion_ICs.txt')
    noisy_IC_index0 = []
    with open(cf, 'r') as fp_cf:
        line = fp_cf.readline().strip() 
        noisy_IC_index0 = [int(index) for index in line.split(',')]
    
    noisy_IC_index = [x - 1 for x in noisy_IC_index0]

    X_noise = melodic_M_echo[:,noisy_IC_index]   ## noisy time course matrix
    ic_file = fullfile(tICA_outDir, 'melodic_IC.nii.gz')
    IC_img = nib.load(ic_file)
    try:
         IC_img_data = IC_img.get_data()   ## get_data() is deprecated in favor of get_fdata() ZSW
    except: 
         IC_img_data = IC_img.get_fdata() 
         
    dimx_ic,dimy_ic,dimz_ic, dim_icN = IC_img_data.shape
    spaceDim_ic = dimx_ic*dimy_ic*dimz_ic
    IC_2d = IC_img_data.reshape((spaceDim_ic, dim_icN))
    IC_2d_transposed = IC_2d.T
    IC_noise = IC_2d_transposed[noisy_IC_index,:]
    Y_noise = np.matmul(X_noise, IC_noise)   ### time x space
    Y_noise_transposed = Y_noise.T  ## space x time
    Y_noise_4D = Y_noise_transposed.reshape((dimx_ic,dimy_ic,dimz_ic, Y_noise_transposed.shape[-1]))
    Y_src = nib.load(src_echo_file)
    ref_data_type = Y_src.get_data_dtype()
    try:
         Y_src_data = Y_src.get_data()   ## get_data() is deprecated in favor of get_fdata() ZSW
    except: 
         Y_src_data = Y_src.get_fdata()  
    Y_clean = Y_src_data - Y_noise_4D
    nifti_clean_image = nib.Nifti1Image(Y_clean.astype(ref_data_type), affine=Y_src.affine, header=Y_src.header)
    nib.save(nifti_clean_image, echo_denoised_file)
    
          
class run_TensorICA(SimpleInterface):
    
    input_spec = run_TensorICAInputSpec
    output_spec = run_TensorICAOutputSpec
    
    def _run_interface(self, runtime):
        
        in_files = self.inputs.in_file
        
        nii_files = ",".join(in_files)
        
        mask_file = self.inputs.mask_file
        
        TR = self.inputs.TR
        
        motion_file = self.inputs.motion_file
        
        bold2mni_matrix_file = self.inputs.bold2mni_matrix_file
        
        echo_N = len(in_files)
       
        out_dir = os.getcwd()
        
        tICA_outDir =  fullfile(out_dir, 'tICA_Out');
        
        ## remove t-mean from each echo before running tensor ICA
        echo1_img = nib.load(in_files[0])
        dimx,dimy,dimz,dimT = echo1_img.shape
        
        echo_dmean_file_list = []   
        echo_global_mean_list = []
        
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
            echo_global_mean_list.append(gMean)
        
        if  Remove_tmean:
            nii_files = ",".join(echo_dmean_file_list)
        
        #melodic -i e1.nii.gz,e2.nii.gz,e3.nii.gz,e4.nii.gz -o tica_out_command --tr=2.47 --mask=mask1.nii.gz -a tica --Oall --Ostats --nobet --mmthresh=0.5 --report
        
        ## run melodic : tensor ICA
        com_str = "melodic -i " + nii_files + " -o " + tICA_outDir + " --tr=" + str(TR) + " --mask=" + mask_file + " -a tica --Oall --Ostats --nobet --mmthresh=0.5 --report"
        #   LOGGER(25, com_str)
        os.system(com_str) 
        
        if Apply_AROMA_to_MergedData:
             ## merge multiple echos into one nii file along time
         
             # fslmerge -t dm_e1234 dm_e1 dm_e2 dm_e3 dm_e4
       
             echos_merged_file = fullfile(tICA_outDir, "echos_merged_file.nii.gz")
             if  Remove_tmean:
                 com_str = "fslmerge -t " + echos_merged_file + " " + " ".join(echo_dmean_file_list)
             else:    
                 com_str = "fslmerge -t " + echos_merged_file + " " + " ".join(in_files)
             #   LOGGER(25, com_str)
                 os.system(com_str)
        
             ## extend motion file for multiple echos 
             single_echo_M = np.loadtxt(motion_file)
             multi_echo_M = np.matlib.repmat(single_echo_M, echo_N, 1)
             multi_echo_motion_file = fullfile(tICA_outDir, "multi_echo_motion.txt")
             np.savetxt(multi_echo_motion_file, multi_echo_M)
        
             ## now run ICA_AROMA.py
        
             # ICA_AROMA.py  -o out -i merged_echos -mc motion_file  -a bold2nii.mat -m mask -tr 2.47 -den nonaggr -dim  0 -md tica_out_dir
             aroma_outDir =  fullfile(out_dir, 'aromaOut');
             if  os.path.exists(aroma_outDir):
                 shutil.rmtree(aroma_outDir)
       
         
             com_str = "ICA_AROMA.py -o " + aroma_outDir + " -i " + echos_merged_file + " -mc " + multi_echo_motion_file
             com_str = com_str  + " -a " + bold2mni_matrix_file + " -m " + mask_file + " -tr " + str(TR)
             com_str = com_str + " -md " + tICA_outDir + " -den nonaggr -dim  0"
             #    LOGGER(25, com_str)
             os.system(com_str)
        
             img = nib.load(in_files[0])
             sx,sy,sz,sT = img.shape
        
             src_denoised_func_file = fullfile(aroma_outDir, "denoised_func_data_nonaggr.nii.gz")
        
             denoised_echo_list = []
       
             for tn in range(echo_N):
                 echo_Out_str = "Out_e" + str(tn+1)
                 echo_tarDir = fullfile(aroma_outDir, echo_Out_str)
                 if not os.path.exists(echo_tarDir):
                     os.makedirs(echo_tarDir)
            
                 echo_denoised_file = fullfile(echo_tarDir, "denoised_func_data_nonaggr.nii.gz")
                 echo_denoised_file_dmean = fullfile(echo_tarDir, "denoised_func_data_nonaggr_dmean.nii.gz")
            
                 beg_num  = str(tn*sT)
                 size_num = str(sT)
            
                 if not Remove_tmean:
                     com_str = "fslroi " + src_denoised_func_file + " " + echo_denoised_file + " " + beg_num + " " + size_num
                 else:
                     com_str = "fslroi " + src_denoised_func_file + " " + echo_denoised_file_dmean + " " + beg_num + " " + size_num
                
                 #      LOGGER(25, com_str)
                 os.system(com_str)
            
                 if Remove_tmean:
                     echo_tmean_str = "e" + str(tn+1) + "_tmean.nii.gz"
                     echo_tmean_file = fullfile(out_dir, echo_tmean_str)
                     if not Use_global_mean:
                         com_str = "fslmaths " + echo_denoised_file_dmean + " -add " + echo_tmean_file + " " + echo_denoised_file
                     else:
                         gMean = echo_global_mean_list[tn]
                         com_str = "fslmaths " + echo_denoised_file_dmean + " -add " + str(gMean) + " " + echo_denoised_file
                     os.system(com_str)
            
                 denoised_echo_list.append(echo_denoised_file)
        
        else:
            ### Apply AROMA to seperated data
           
            denoised_echo_list = []
            melodic_mix_file_all_echos_backup = fullfile(tICA_outDir, 'melodic_mix_orig')
            melodic_mix_file_all_echos = fullfile(tICA_outDir, 'melodic_mix')
            shutil.copyfile(melodic_mix_file_all_echos, melodic_mix_file_all_echos_backup)
            melodic_M_all = np.loadtxt(melodic_mix_file_all_echos)
            M_rowL, M_colL = melodic_M_all.shape
            M_tN = int(M_rowL/echo_N)
            
            aroma_outDir =  fullfile(out_dir, 'aromaOut');
            if  os.path.exists(aroma_outDir):
                 shutil.rmtree(aroma_outDir)
                 
            for tn in range(echo_N):
                 echo_Out_str = "Out_e" + str(tn+1)
                 echo_tarDir = fullfile(aroma_outDir, echo_Out_str)
                 if os.path.exists(echo_tarDir):
                     shutil.rmtree(echo_tarDir)
            
                 if  Remove_tmean:
                     src_echo_file = echo_dmean_file_list[tn]
                 else:
                     src_echo_file = in_files[tn]
                     
                 ## generate mix_matrix file for each echo
                 
                 melodic_M_echo = melodic_M_all[tn*M_tN:(tn+1)*M_tN,:]
              
                 melodic_mix_file_echo = fullfile(tICA_outDir, 'melodic_mix' + str(tn+1))
                 np.savetxt(melodic_mix_file_echo, melodic_M_echo)
                 shutil.copyfile(melodic_mix_file_echo, melodic_mix_file_all_echos)
             
                 if Apply_AROMA_to_Echo1_Only and tn > 0:
                      ## fsl_regfilt -i e3.nii.gz -d tICA_out/melodic_mix3 -o  denoised_e3 -f "$(< armaOut/classified_motion_ICs.txt)"
                      if not os.path.exists(echo_tarDir):
                          os.makedirs(echo_tarDir)
                      echo1_tarDir = fullfile(aroma_outDir, "Out_e1")
                      classified_motion_ICs_file = fullfile(echo1_tarDir, "classified_motion_ICs.txt")
                     ## classified_motion_ICs_str = "$(< " +  classified_motion_ICs_file + ")"
                      with open(classified_motion_ICs_file) as ft:
                           noisyICs = ft.readline()
                      echo_denoised_file = fullfile(echo_tarDir, "denoised_func_data_nonaggr.nii.gz")
                      com_str = "fsl_regfilt -i " + src_echo_file + " -d " + melodic_mix_file_echo + " -o " + echo_denoised_file + " -f " + '\"' + noisyICs + '\"'
                      os.system(com_str)
                      if Use_OuterProduct_for_denosing:  ## redo denoising
                          TensorICA_Denoise_UseOutProduct(echo_tarDir, tICA_outDir, melodic_M_echo, src_echo_file, classified_motion_ICs_file)
                 else:

                      com_str = "ICA_AROMA.py -o " + echo_tarDir + " -i " + src_echo_file + " -mc " + motion_file
                      com_str = com_str  + " -a " + bold2mni_matrix_file + " -m " + mask_file + " -tr " + str(TR)
                      com_str = com_str + " -md " + tICA_outDir + " -den nonaggr -dim  0"
                      #    LOGGER(25, com_str)
                      os.system(com_str)
                      echo_denoised_file = fullfile(echo_tarDir, "denoised_func_data_nonaggr.nii.gz")
                      
                      if Use_OuterProduct_for_denosing:  ## redo denoising
                           classified_motion_ICs_file = fullfile(echo_tarDir, "classified_motion_ICs.txt")
                           TensorICA_Denoise_UseOutProduct(echo_tarDir, tICA_outDir, melodic_M_echo, src_echo_file, classified_motion_ICs_file)
                      
                 if  Remove_tmean:
                  
                     echo_denoised_file_dmean = fullfile(echo_tarDir, "denoised_func_data_nonaggr_dmean.nii.gz")
                     
                     try:
                         shutil.movefile(echo_denoised_file, echo_denoised_file_dmean)
                     except:
                         shutil.move(echo_denoised_file, echo_denoised_file_dmean)    
                    
                     echo_tmean_str = "e" + str(tn+1) + "_tmean.nii.gz"
                     echo_tmean_file = fullfile(out_dir, echo_tmean_str)
                     
                     if not Use_global_mean:
                         com_str = "fslmaths " + echo_denoised_file_dmean + " -add " + echo_tmean_file + " " + echo_denoised_file
                     else:
                         gMean = echo_global_mean_list[tn]
                         com_str = "fslmaths " + echo_denoised_file_dmean + " -add " + str(gMean) + " " + echo_denoised_file
                     os.system(com_str)
                     
                 
                 denoised_echo_list.append(echo_denoised_file)
                 
            
        self._results['denoised_func_files'] = denoised_echo_list
                     
        
        return runtime
        
      
        
        