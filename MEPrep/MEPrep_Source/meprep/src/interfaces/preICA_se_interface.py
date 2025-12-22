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
import shutil
from pathlib import Path

from os.path import join as fullfile

from niworkflows.utils.connections import  listify

from nipype.interfaces.base import OutputMultiPath


from nipype import logging
from nipype.interfaces.base import (
    traits, TraitedSpec, File,
    CommandLine, CommandLineInputSpec)

LOGGER = logging.getLogger('nipype.interface')

doDEBUG=False

doDEBUG1=False

doDEBUG2 = False


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


#ICA_AROMA.py -o $PWD/out1 -i $PWD/e1.nii.gz -mc $PWD/motion_params.txt  -m $PWD/average_mask.nii.gz  -tr 1.6  -den nonaggr -np -dim 0
class preICAInputSpec(CommandLineInputSpec):
    
    out_dir = traits.List(File(exists=False),
                           ['.'],
                           argstr='-o %s',
                           position=1,
                           usedefault=True,
                           minlen=1,
                           desc='Output dir of the preICA results')
     
    in_file = traits.File(exists=True,
                           argstr='-i %s',
                           position=2,
                           mandatory=True,
                           minlen=1,
                           desc='One of multi-echo BOLD EPIs')
    
    mask_file = traits.File(exists=True,
                           argstr='-m %s',
                           position=3,
                           mandatory=True,
                           minlen=1,
                           desc='Mask file to mask multi-echo BOLD EPIs')
    
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
    
    

class preICAOutputSpec(TraitedSpec):
     denoised_func_files = OutputMultiPath(File(exists=True), desc='denoised EPI file, i.e.,denoised_func_data_nonaggr.nii.gz')
     bold_mask_file = OutputMultiPath(File(exists=True), desc='bold mask file, i.e.,average_mask.nii.gz')
     denoising_type_selected = traits.Str('nonaggr',
                                          desc = "selected of denosiing types"
         )
   

class preICA(CommandLine):
    """
    Runs AROMA-ICA to denoise each echo data before calling t2smap to estimate T2* map and create
    an optimally combined ME-EPI time series.
    Our validation results indicate that preICA-denoising each echo can signiifcantly improve
    exponential model fitting and increase tSNR and test-retest reliability of functional connectome

    Example
    =======
    >>> from fmriprep.interfaces import preICA_interface
    >>> preICA = preICA_interface.preICA()
    >>> preICA.inputs.in_files = ['sub-01_run-01_echo-1_bold.nii.gz', \
                                  'sub-01_run-01_echo-2_bold.nii.gz', \
                                  'sub-01_run-01_echo-3_bold.nii.gz']
    
    """
    
    _cmd = 'ICA_AROMA.py'  #'aroma_debug'   #'ICA_AROMA.py'
    
    if doDEBUG1:
        _cwd = 'aroma_debug' 
    
    input_spec = preICAInputSpec
    output_spec = preICAOutputSpec

    current_echo = 0
    current_echo_bold_file = None
    full_out_dir = ""
    bold_mask_file = []
    bold_path = ""
    
    _denoising_type = ""
    
    def _format_arg(self, name, trait_spec, value):
        if name == 'in_file':
            bold_file = value 
            bold_str = ''.join(bold_file)
            entities = extract_entities_from_base(bold_file)
            echo_idxs = listify(entities.get("echo", []))
            if not echo_idxs:
                idx = bold_str.find("_echoidx_")+len("_echoidx_")
                echo_idxs0 = int(bold_str[idx])+1
                echo_idxs  = [str(echo_idxs0)]
                
            self.current_echo = echo_idxs
            self.current_echo_bold_file = value
            self.bold_path = os.path.dirname(os.path.abspath(bold_str))
        if name == 'mask_file':
            self.bold_mask_file = value
        if name == 'bold2mni_matrix_file':
            matrix_file = value
            print("BOLD2MNI matrix file is %s " % matrix_file)
        if name == 'TR'    :
            print("TR is %.10f" % value)
        if name == 'denoising_type':
            dntype = value
            print("The denoising type is %s" % dntype)
            self._denoising_type = dntype
        if name == 'out_dir':
            echo_out = "out_echo" + str(self.current_echo[0])
            
            # self.full_out_dir = fullfile(value[0],echo_out)
            self.full_out_dir  = fullfile(self.bold_path,echo_out, self._denoising_type)
            outDir = Path(self.full_out_dir)
            if outDir.is_dir():
                print("The folder %s exists, now we remove it ... " % outDir)
                if doDEBUG1==False:
                   shutil.rmtree(outDir)
                
            out_dir_wf = os.getcwd()
            print("the cruurent dir before calling preICA is %s" % out_dir_wf)
            #self.full_out_dir = fullfile(out_dir_wf,echo_out)
            
            value = [self.full_out_dir]
           
                   
        return super(preICA, self)._format_arg(name, trait_spec, value)
      

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_dir_wf = os.getcwd()
        print("the cruurent dir after calling preICA is %s" % out_dir_wf)
        out_dir_aroma = self.full_out_dir
        current_denoised_func_file = os.path.join(out_dir_aroma, 'denoised_func_data_%s.nii.gz' %self._denoising_type)
        if doDEBUG2:
            current_denoised_func_file = os.path.join(out_dir_aroma, 'denoised_text_desc-%s.tsv' %self._denoising_type)
            with open(current_denoised_func_file, 'w') as f:
                 f.write("the current denosing type is %s " % self._denoising_type)
                 f.write('\r\n')
                 f.write("the current echo is %s " %self.current_echo)
                 f.write('\r\n')
       
        outputs['denoised_func_files'] = current_denoised_func_file 
        outputs['bold_mask_file'] = ''.join(self.bold_mask_file)
        outputs['denoising_type_selected'] = self._denoising_type
       
        return outputs

