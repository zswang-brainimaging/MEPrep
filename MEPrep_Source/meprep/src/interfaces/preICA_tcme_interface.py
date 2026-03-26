"""
Multi-echo Interface to implement optimally combining, 
denosing with ME-ICA and both functons

Made by Dr. Zhishun Wang in 2021, 2022

This is the interface to Run temporally concatenated ME ICA (tcme-ICA)

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

Remove_tmean = True
Use_global_mean = False

class run_tcmeICAInputSpec(BaseInterfaceInputSpec):
    
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

class run_tcmeICAOutputSpec(TraitedSpec):
   denoised_func_files = OutputMultiPath(File(exists=True), desc='denoised EPI file, i.e.,denoised_func_data_nonaggr.nii.gz')
  
           
      
class run_tcmeICA(SimpleInterface):
    
    input_spec = run_tcmeICAInputSpec
    output_spec = run_tcmeICAOutputSpec
    
    def _run_interface(self, runtime):
        
        in_files = self.inputs.in_file
        
        mask_file = self.inputs.mask_file
        
        TR = self.inputs.TR
        
        motion_file = self.inputs.motion_file
        
        bold2mni_matrix_file = self.inputs.bold2mni_matrix_file
        
        echo_N = len(in_files)
        
        out_dir = os.getcwd()
        
        tcmeICA_outDir =  fullfile(out_dir, 'tcmeICA_Out');
        if not os.path.exists(tcmeICA_outDir):
             os.makedirs(tcmeICA_outDir)
             
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
                 
        
        ## merge multiple echos into one nii file along time
        
       # fslmerge -t dm_e1234 dm_e1 dm_e2 dm_e3 dm_e4
       
        echos_merged_file = fullfile(tcmeICA_outDir, "echos_merged_file.nii.gz")
        if not Remove_tmean:
            com_str = "fslmerge -t " + echos_merged_file + " " + " ".join(in_files)
        else:
            com_str = "fslmerge -t " + echos_merged_file + " " + " ".join(echo_dmean_file_list)
        os.system(com_str)
        
        #melodic -i e1.nii.gz,e2.nii.gz,e3.nii.gz,e4.nii.gz -o tica_out_command --tr=2.47 --mask=mask1.nii.gz -a tica --Oall --Ostats --nobet --mmthresh=0.5 --report
        
        ## run melodic : tensor ICA
        com_str = "melodic -i " + echos_merged_file + " -o " + tcmeICA_outDir + " --tr=" + str(TR) + " --mask=" + mask_file + " --Oall --Ostats --nobet --mmthresh=0.5 --report"
        os.system(com_str) 
        
               
        ## extend motion file for multiple echos 
        single_echo_M = np.loadtxt(motion_file)
        multi_echo_M = np.matlib.repmat(single_echo_M, echo_N, 1)
        multi_echo_motion_file = fullfile(tcmeICA_outDir, "multi_echo_motion.txt")
        np.savetxt(multi_echo_motion_file, multi_echo_M)
        
        ## now run ICA_AROMA.py
        
        # ICA_AROMA.py  -o out -i merged_echos -mc motion_file  -a bold2nii.mat -m mask -tr 2.47 -den nonaggr -dim  0 -md tica_out_dir
        aroma_outDir =  fullfile(out_dir, 'aromaOut');
        if os.path.exists(aroma_outDir):
             shutil.rmtree(aroma_outDir)
       
         
        com_str = "ICA_AROMA.py -o " + aroma_outDir + " -i " + echos_merged_file + " -mc " + multi_echo_motion_file
        com_str = com_str  + " -a " + bold2mni_matrix_file + " -m " + mask_file + " -tr " + str(TR)
        com_str = com_str + " -md " + tcmeICA_outDir + " -den nonaggr -dim  0"
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
            
            echo_fslroi_file = fullfile(echo_tarDir, "denoised_func_fslroi.nii.gz")
            echo_denoised_file = fullfile(echo_tarDir, "denoised_func_data_nonaggr.nii.gz")
            
            beg_num  = str(tn*sT)
            size_num = str(sT)
            
            if not Remove_tmean:
               com_str = "fslroi " + src_denoised_func_file + " " + echo_denoised_file + " " + beg_num + " " + size_num
               os.system(com_str)
            else:
               com_str = "fslroi " + src_denoised_func_file + " " + echo_fslroi_file + " " + beg_num + " " + size_num
               os.system(com_str)
               echo_tmean_str = "e" + str(tn+1) + "_tmean.nii.gz"
               echo_tmean_file = fullfile(out_dir, echo_tmean_str)
               if not Use_global_mean:
                  com_str = "fslmaths " + echo_fslroi_file + " -add " + echo_tmean_file + " " + echo_denoised_file
               else:
                   gMean = echo_global_mean_list[tn]
                   com_str = "fslmaths " + echo_fslroi_file + " -add " + str(gMean) + " " + echo_denoised_file
                   
               os.system(com_str)
            
            denoised_echo_list.append(echo_denoised_file)
        
        
        self._results['denoised_func_files'] = denoised_echo_list
                     
        
        return runtime
        
      
        
        