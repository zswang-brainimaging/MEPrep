#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 23:11:59 2021

@author: zswang
"""
'''
This is an interface code made by Dr. Zhishun Wang to implement the preICA based on AROMA-ICA algorithm
to denoise multi-echo data before combinning them
'''
import os

from os.path import join as fullfile

from nipype import logging
from nipype.interfaces.base import (
    traits, TraitedSpec, File,
    CommandLine, CommandLineInputSpec)

LOGGER = logging.getLogger('nipype.interface')

#"fslmaths %s -mas %s %s" % (echo_file, mask_file, masked_bold)

doDEBUGa = False

class apply_maskInputSpec(CommandLineInputSpec):
    
    
    in_file = traits.File(exists=True,
                           argstr=' %s',
                           position=1,
                           mandatory=True,
                           minlen=1,
                           desc='One of multi-echo BOLD EPIs')
    
    mask_file = traits.File(exists=True,
                           argstr='-mas %s',
                           position=2,
                           mandatory=True,
                           minlen=1,
                           desc='Mask file to mask multi-echo BOLD EPIs')
    
   
    
    out_file = traits.List(File(exists=False),
                           'masked_bold.nii.gz',
                           argstr=' %s',
                           position=3,
                           mandatory=True,
                           minlen=1,
                           usedefault=True,
                           desc='One of masked multi-echo BOLD EPIs') 
    

class apply_maskOutputSpec(TraitedSpec):
   
     masked_bold = File(exists=True, desc='bold masked by a mask, i.e.,average_mask.nii.gz')
   #  mask_file = File(exists=True, desc='the mask file used to mask bold, i.e.,average_mask.nii.gz')
   

class apply_mask(CommandLine):
    """
    Apply a mask to one of multiple echo files
    
    """
    
    _cmd = 'fslmaths'  
    
    if doDEBUGa:
        _cmd = 'aroma_debug' 
    
    input_spec = apply_maskInputSpec
    output_spec = apply_maskOutputSpec

    
    masked_bold_file = ""
    bold_path = ""
    bold_fname = ""
    mask_file = ""
    
   # denoised_func_file_list = []
    
    def _format_arg(self, name, trait_spec, value):
        if name == 'in_file':
            bold_file = value 
            bold_str = ''.join(bold_file)
            print("bold file inputted into apply mask is: %s" % value)
            
            self.bold_path = os.path.dirname(os.path.abspath(bold_str))
            self.bold_fname = os.path.basename(os.path.abspath(bold_str))
            
        if name == 'mask_file':
            self.mask_file = ''.join(value[0])
            print("mask file inputted into apply mask is: %s" % value)
        if name == 'out_file':
            
            self.masked_bold_file = fullfile(self.bold_path, "masked_" + self.bold_fname)  
            
            if os.path.exists(self.masked_bold_file):
                if doDEBUGa==False:
                   os.remove(self.masked_bold_file)
            else:
                 print("Can not delete the masked file as it doesn't exists")
            
            value = [self.masked_bold_file]
           
              
        return super(apply_mask, self)._format_arg(name, trait_spec, value)
      

    def _list_outputs(self):
        outputs = self._outputs().get()
      
        outputs['masked_bold'] = self.masked_bold_file
    #    outputs['mask_file'] = self.mask_file
       
        return outputs

