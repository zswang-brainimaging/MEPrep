#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 23:11:59 2021

@author: zswang
"""
'''
This is an interface code made by Dr. Zhishun Wang to implement the preICA based on AROMA-ICA algorithm
to denoise multi-echo data before combinning them

This is to wrap FSL linear registration : func to MNI152 template

'''
import os

from os.path import join as fullfile

from nipype import logging
from nipype.interfaces.base import (
    traits, TraitedSpec, File,
    CommandLine, CommandLineInputSpec)

LOGGER = logging.getLogger('nipype.interface')

doDEBUGa = False

class registerInputSpec(CommandLineInputSpec):
    
    
    in_file = traits.File(exists=True,
                           argstr='-in %s',
                           position=1,
                           mandatory=True,
                           minlen=1,
                           desc='One of multi-echo BOLD EPIs')
    
    ref_file = traits.File(exists=False,
                           argstr='-ref %s',
                           position=2,
                           mandatory=True,
                           minlen=1,
                           desc='Reference file (MNI152) for ME-EPI to coregister to ')
    
    out_file = traits.File(exists=False,
                           argstr='-omat %s',
                           position=3,
                           mandatory=True,
                           minlen=1,
                           desc='Affine transform file for registering multi-echo BOLD EPIs to MNI152') 
    

class registerOutputSpec(TraitedSpec):
   
     bold2mni_matrix_file = File(exists=True, desc='bold masked by a mask, i.e.,average_mask.nii.gz')
  
## wrarp flirt [options] -in <inputvol> -ref <refvol> -omat <outputmatrix>
###
class register(CommandLine):
    """
    Apply a mask to one of multiple echo files
    
    """
    
    _cmd = 'flirt'  
    
    if doDEBUGa:
        _cmd = 'aroma_debug' 
    
    input_spec = registerInputSpec
    output_spec = registerOutputSpec

    
    affine_matrix_file = ""
    
   # denoised_func_file_list = []
       
    def _format_arg(self, name, trait_spec, value):
        if name == 'in_file':
            bold_file = value
            bold_str = ''.join(bold_file)
            print("bold file inputted into apply mask is: %s" % value)
            out_dir = os.getcwd()
            bold_tmean_file = fullfile(out_dir,"bold_echo1_tmean.nii.gz")
            com_str = "fslmaths " + bold_str + " -Tmean " + bold_tmean_file
            os.system(com_str)
            com_file = fullfile(out_dir, "tmean_command.txt")
            with open(com_file, 'w') as fp:
                fp.write(com_str)
            
            value = bold_tmean_file
            
                       
        if name == 'ref_file':
            if not ".nii" in value or not os.path.exists(value):
               fslDir = os.environ["FSLDIR"]
               refvol = fullfile(fslDir, "data", "standard", "MNI152_T1_2mm_brain.nii.gz")
               value = refvol
            
        if name == 'out_file':
            
           out_dir = os.getcwd()
           self.affine_matrix_file =fullfile(out_dir, "bold_tmean2mni_affine_matrix.mat")
           value = self.affine_matrix_file
           
           
              
        return super(register, self)._format_arg(name, trait_spec, value)
      

    def _list_outputs(self):
        outputs = self._outputs().get()
      
        outputs['bold2mni_matrix_file'] = self.affine_matrix_file
   
       
        return outputs

