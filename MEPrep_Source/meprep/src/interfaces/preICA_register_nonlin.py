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

from niworkflows.utils.bids import collect_data  ##ZSW
from fmriprep import config   ##ZSW

from nipype import logging
from nipype.interfaces.base import (
    traits, TraitedSpec, File,
    CommandLine, CommandLineInputSpec)

LOGGER = logging.getLogger('nipype.interface')

doDEBUGa = False

Register_Method_Indirect = True

def extract_entities_from_base(file_list):
    """
    Return a dictionary of common entities given a list of files.

    Examples
    --------
    >>> extract_entities("sub-01/anat/sub-01_T1w.nii.gz")
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_T1w.nii.gz"] * 2)
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_run-1_T1w.nii.gz",
    ...                   "sub-01/anat/sub-01_run-2_T1w.nii.gz"])
    {'subject': '01', 'run': [1, 2], 'suffix': 'T1w', 'datatype': 'anat',
     'extension': '.nii.gz'}

    """
    from collections import defaultdict
    from bids.layout import parse_file_entities
    from niworkflows.utils.connections import  listify

    entities = defaultdict(list)
    for e, v in [
        ev_pair
        for f in listify(file_list)
        for ev_pair in parse_file_entities(f).items()
    ]:
        entities[e].append(v)

    def _unique(inlist):
        inlist = sorted(set(inlist))
        if len(inlist) == 1:
            return inlist[0]
        return inlist

    return {k: _unique(v) for k, v in entities.items()}


class registerInputSpec(CommandLineInputSpec):
    
    
    in_file = traits.File(exists=True,
                           argstr='--in=%s',
                           position=1,
                           mandatory=True,
                           minlen=1,
                           desc='One of multi-echo BOLD EPIs')
    
    ref_file = traits.File(exists=False,
                           argstr='--ref=%s',
                           position=2,
                           mandatory=True,
                           minlen=1,
                           desc='Reference file (MNI152) for ME-EPI to coregister to ')
    
    out_file = traits.File(exists=False,
                           argstr='--cout=%s',
                           position=3,
                           mandatory=True,
                           minlen=1,
                           desc='Nonlinear transform warp file for registering multi-echo BOLD EPIs to MNI152') 
    

class registerOutputSpec(TraitedSpec):
   
     bold2mni_matrix_file = File(exists=True, desc='bold masked by a mask, i.e.,average_mask.nii.gz')
  
## wrarp flirt [options] -in <inputvol> -ref <refvol> -omat <outputmatrix>
###
class register(CommandLine):
    """
    Apply a mask to one of multiple echo files
    
    """
    
    _cmd = 'fnirt'  
    
    if doDEBUGa:
        _cmd = 'aroma_debug' 
    
    input_spec = registerInputSpec
    output_spec = registerOutputSpec

    
    affine_matrix_file = ""
    nonliner_warp_file  = ""
    
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
            
            if not Register_Method_Indirect:
                value = bold_tmean_file
            
            else:
            
                 bold_source_file = config.execution.bold_source_file
                 if isinstance(bold_source_file, list) or isinstance(bold_source_file, str):
                     EN = extract_entities_from_base(bold_source_file)
                     subject_id = EN['subject']
                 else:
                     subject_id = config.execution.participant_label[0]
                     
                 subject_data = collect_data(
                           config.execution.layout,
                           subject_id,
                           config.execution.task_id,
                           config.execution.echo_idx,
                           bids_filters=config.execution.bids_filters)[0]
                     
                 src_T1w = ''.join(subject_data['t1w'])
                 tar_T1w_brain = fullfile(out_dir,"T1w_brain.nii.gz")
                 com_str = "bet " + src_T1w + " " + tar_T1w_brain
                 os.system(com_str)
                 reg_T1w_brain = fullfile(out_dir,"T1w_brain2bold.nii.gz")
            
                 com_str = "flirt  -in " + tar_T1w_brain + " -ref  " + bold_tmean_file + " -out " + reg_T1w_brain
                 os.system(com_str)
                 
                             
                 fslDir = os.environ["FSLDIR"]
                 refvol = fullfile(fslDir, "data", "standard", "MNI152_T1_2mm_brain.nii.gz")
                  ## register reg_T1w_brain to MNI152
                 T1w2bold2std_nii = fullfile(out_dir, "T1w2bold2std.nii.gz")             
                 T1w2bold2std_affine_matrix_file = fullfile(out_dir, "T1w2bold2std_affine.mat")
                 com_str = "flirt  -in " + reg_T1w_brain + " -ref  " + refvol + " -out " + T1w2bold2std_nii  + " -omat "+ T1w2bold2std_affine_matrix_file
                 os.system(com_str)
                 self.affine_matrix_file = T1w2bold2std_affine_matrix_file 
               
                 value = T1w2bold2std_nii
            
        if name == 'ref_file':
            if not ".nii" in value or not os.path.exists(value):
               fslDir = os.environ["FSLDIR"]
               refvol = fullfile(fslDir, "data", "standard", "MNI152_T1_2mm_brain.nii.gz")
             
               value = refvol
            
        if name == 'out_file':
            
           out_dir = os.getcwd()
           self.nonliner_warp_file =fullfile(out_dir, "T1w2bold2std_warp.nii.gz")
           value = self.nonliner_warp_file
           
           
              
        return super(register, self)._format_arg(name, trait_spec, value)
      

    def _list_outputs(self):
        outputs = self._outputs().get()
      
        bold2std_register_list = []
        bold2std_register_list.append(self.affine_matrix_file)
        bold2std_register_list.append(self.nonliner_warp_file)
        
        outputs['bold2mni_matrix_file'] = bold2std_register_list
   
       
        return outputs

