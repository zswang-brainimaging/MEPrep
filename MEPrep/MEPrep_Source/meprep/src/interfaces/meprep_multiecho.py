"""
Multi-echo Interface to implement optimally combining, 
denosing with ME-ICA and both functons

Made by Dr. Zhishun Wang in 2021, 2022


"""
Generate_Residual_Rsquare_Maps = False    ##ZSW Since preICA has been validated to reduce the fitting residuals in previous versions of MEPrep 
                                          ## pipleines, and users can use confounds file to do QC,  the functionality of generating these metrics
                                          ## may be no longer needed to newer versions of MEPrep, ZSW

import os
from os.path import join as fullfile

import logging

from threadpoolctl import threadpool_limits

if not Generate_Residual_Rsquare_Maps:
    from tedana.workflows.tedana import tedana_workflow
    from tedana.workflows.t2smap import t2smap_workflow
    from tedana.workflows.tedana import _get_parser as tedana_get_parser
    from tedana.workflows.t2smap import _get_parser as t2smap_get_parser
else:
    from tedana.workflows.tedana_residual import tedana_workflow
    from tedana.workflows.t2smap_residual import t2smap_workflow
    from tedana.workflows.tedana_residual import _get_parser as tedana_get_parser
    from tedana.workflows.t2smap_residual import _get_parser as t2smap_get_parser

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from fmriprep.config import DEFAULT_MEMORY_MIN_GB
from fmriprep.interfaces import DerivativesDataSink

from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File,
    SimpleInterface
)

from fmriprep import config

import shlex

import shutil

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')

LOGGER = logging.getLogger('nipype.interface')

Output_Fitting_Data =  True

class run_tedanaInputSpec(BaseInterfaceInputSpec):
    
     in_files = traits.List(File(exists=True),
                           argstr='-d %s',
                           position=1,
                           mandatory=True,
                           minlen=3,
                           desc='multi-echo BOLD EPIs')
     
     echo_times = traits.List(traits.Float,
                             argstr='-e %s',
                             position=2,
                             mandatory=True,
                             minlen=3,
                             desc='echo times')
    
     mask_file = traits.File(exists=True,
                           argstr='--mask %s',
                           position=3,
                           mandatory=True,
                           minlen=1,
                           desc='EPI mask file')
     
     fittype = traits.Enum('curvefit', 'loglin',
                          argstr='--fittype %s',
                          position=4,
                          usedefault=True,
                          desc=('Desired fitting method: '
                                '"loglin" means that a linear model is fit '
                                'to the log of the data. '
                                '"curvefit" means that a more computationally '
                                'demanding monoexponential model is fit '
                                'to the raw data.'))
    
     tedana_choice = traits.Enum ('optcom', 'optcomDenoised', 'echo2',
                          argstr='--tedana_choice %s',
                          position=5,
                          usedefault=True,
                          desc=('Choices of tedana methods: '
                                'optcom: use t2smap workflow from tedana package to optimally combine multi-echo data;',
                                'optcomDenoised: use ME-ICA workflow from tedana package to run ME-ICA on the optimally combined multi-echo data to denoise.',
                                'echo2: output echo2 data for comparisons with multi-echo data',
                                'default is optcom, which is the default and only option in the original fmriprep pipeline.'))
     source_file = traits.List(File(exists=False),
                           argstr=' %s',
                           position=6,
                        #   mandatory=True,
                        #   minlen=1,
                           desc='bold source files for data sink')
     preDenoising_choice = traits.Enum ('Raw', 'preICA_scme',
                          argstr='--preDenoising_choice %s',
                          position=7,
                          usedefault=True,
                          desc=('Choices of preDenoising methods: '
                                'Raw: No preDenoising (preICA);',
                                'preICA_scme: use PICA to predenoise ME-fMRI data before OptCom.',
                                'default is Raw.'))

class run_tedanaOutputSpec(TraitedSpec):
      T2starmap = File(exists=False, desc='Full estimated T2* 3D map')
      S0map = File(exists=False, desc='Full S0 3D map')
      
      if Generate_Residual_Rsquare_Maps:
         Residualmap = File(exists=False, desc='Full estimated model fitting residual 3D map')
         Rsquaremap = File(exists=False, desc='Full estimated model fitting R-square 3D map')
      
      optcom_bold = File(exists=True, desc='optimally combined ME-EPI time series')
      optcomDenoised_bold = File(exists=True, desc='ME-ICA-denoised optimally combined ME-EPI time series')
      tedana_choice_selected = traits.Str('optcom', desc = "selected tedana choice")
      tedana_output_bold = File(exists=True, desc='optimally combined or Denoised & optimally combined ME-EPI time series')
  #    tedana_choice_model_fitting_desc = traits.Str('optcom', desc = "selected tedana choice")
      
      
      
class run_tedana(SimpleInterface):
    
    input_spec = run_tedanaInputSpec
    output_spec = run_tedanaOutputSpec
    
    def _run_interface(self, runtime):
        
        in_files = self.inputs.in_files
        value = self.inputs.echo_times
        value = [te * 1000 for te in value]
        echo_times = value
        mask_file = self.inputs.mask_file
        fittype   = self.inputs.fittype
        tedana_choice = self.inputs.tedana_choice
        source_file = self.inputs.source_file
        
        if not isinstance(source_file, list) and not isinstance(source_file, str):
            source_file = config.execution.bold_source_file
        
               
        input_str = "-d %s " % " ".join(in_files) + "--fittype %s " % fittype
        input_str = input_str + "-e %s " % " ".join([str(it) for it in echo_times]) 
        input_str = input_str + "--mask %s " % mask_file 
        
        
        optv = shlex.split(input_str)
        
        if tedana_choice == "echo2" and not Output_Fitting_Data:
            
            out_dir = os.getcwd()
            src_file = in_files[1]
            tar_file = fullfile(out_dir, "echo2.nii.gz")
            com_str = "cp -p " + src_file + " " + tar_file
            com_file = fullfile(out_dir, "command.txt")
            with open(com_file, 'w') as fp:
                 fp.write(com_str)
            
            shutil.copyfile(src_file, tar_file)    
            self._results['tedana_output_bold'] = tar_file  ## echo2 data ZSW
            self._results['tedana_choice_selected'] = tedana_choice
            
         
                     
            return runtime
            
        if tedana_choice =="optcomDenoised":
            options = tedana_get_parser().parse_args(optv)
        else:
            options = t2smap_get_parser().parse_args(optv)
           
        
        kwargs = vars(options)
        n_threads = kwargs.pop('n_threads')
        n_threads = None if n_threads == -1 else n_threads
        with threadpool_limits(limits=n_threads, user_api=None):
           if tedana_choice =="optcomDenoised":
              tedana_workflow(**kwargs)
              out_dir = os.getcwd()
              if isinstance(source_file, list) or isinstance(source_file, str):
                   entities = extract_entities_from_base(source_file)
                   fmriprep_dir = str(config.execution.fmriprep_dir)
                   
                   #T2starmap_dict = {'T2starmap': fullfile(out_dir, "T2starmap.nii.gz"),
                   #                  'S0map': fullfile(out_dir, "S0map.nii.gz"),
                   #                  'Residualmap': fullfile(out_dir, "Residualmap.nii.gz"),
                   #                  'Rsquaremap': fullfile(out_dir, "Rsquaremap.nii.gz")
                   #                  }
                   ##  output_model_fitting_data(T2starmap_dict, fmriprep_dir, source_file, entities, "Use_DerivativesDataSink")
                   
                   bold_source_info_dict = entities
                   bold_source_info_dict["source_file"] = source_file
                   bold_source_info_dict["fmriprep_dir"] = fmriprep_dir
                   bold_source_info_dict["T2starmap"] = fullfile(out_dir, "T2starmap.nii.gz")
                   csv_file = fullfile(os.getcwd(), "bold_source_info.csv")
                   with open(csv_file, 'w') as fp:
                       for key in bold_source_info_dict.keys():
                           fp.write("%s,%s\n"%(key,bold_source_info_dict[key]))
                         
                   
              
              com_str = "tedana " + input_str
              com_file = fullfile(out_dir, "command.txt")
              with open(com_file, 'w') as fp:
                   fp.write(com_str)
              self._results['optcom_bold'] = fullfile(out_dir, "desc-optcom_bold.nii.gz")
              
              denoised_bold_file = fullfile(out_dir, "desc-optcomDenoised_bold.nii.gz")  ## older version ME-ICA
              if not os.path.exists(denoised_bold_file):
                  denoised_bold_file = fullfile(out_dir, "desc-denoised_bold.nii.gz")  ## newer version ME-ICA
              if not os.path.exists(denoised_bold_file):
                  LOGGER.log(25, f'ME-ICA may not have been run successfully to generate denoised BOLD in {out_dir} !')
              
              self._results['optcomDenoised_bold'] = denoised_bold_file
             
              self._results['T2starmap'] = fullfile(out_dir, "T2starmap.nii.gz")
              self._results['S0map'] = fullfile(out_dir, "S0map.nii.gz")
              
              if Generate_Residual_Rsquare_Maps:
                  self._results['Residualmap'] = fullfile(out_dir, "Residualmap.nii.gz")
                  self._results['Rsquaremap'] = fullfile(out_dir, "Rsquaremap.nii.gz")
              
              self._results['tedana_output_bold'] = denoised_bold_file
              self._results['tedana_choice_selected'] = tedana_choice
              
             
           else:
            #  sub_dict_keys = ['data', 'tes','out_dir', 'mask', 'fittype', 'fitmode', 'combmode', 'debug', 'quiet'] 
            #   kwargs_subset = {key: kwargs[key] for key in sub_dict_keys}
              t2smap_workflow(**kwargs)  #_subset)
              out_dir = os.getcwd()
              if isinstance(source_file, list) or isinstance(source_file, str):
                   entities = extract_entities_from_base(source_file)
                   fmriprep_dir = str(config.execution.fmriprep_dir)
                   
                   #T2starmap_dict = {'T2starmap': fullfile(out_dir, "T2starmap.nii.gz"),
                   #                  'S0map': fullfile(out_dir, "S0map.nii.gz"),
                   #                  'Residualmap': fullfile(out_dir, "Residualmap.nii.gz"),
                   #                  'Rsquaremap': fullfile(out_dir, "Rsquaremap.nii.gz")
                   #                  }
                  
                   ## output_model_fitting_data(T2starmap_dict, fmriprep_dir, source_file, entities, "Use_DerivativesDataSink")
                   
                   entities = extract_entities_from_base(source_file)
                   fmriprep_dir = str(config.execution.fmriprep_dir)
                   bold_source_info_dict = entities
                   bold_source_info_dict["source_file"] = source_file
                   bold_source_info_dict["fmriprep_dir"] = fmriprep_dir
                   bold_source_info_dict["T2starmap"] = fullfile(out_dir, "T2starmap.nii.gz")
                   csv_file = fullfile(os.getcwd(), "bold_source_info.csv")
                   with open(csv_file, 'w') as fp:
                       for key in bold_source_info_dict.keys():
                           fp.write("%s,%s\n"%(key,bold_source_info_dict[key]))
                   
              com_str = "t2smap " + input_str
              com_file = fullfile(out_dir, "command.txt")
              with open(com_file, 'w') as fp:
                   fp.write(com_str)
              self._results['optcom_bold'] = fullfile(out_dir, "desc-optcom_bold.nii.gz")
              
              self._results['T2starmap'] = fullfile(out_dir, "T2starmap.nii.gz")
              self._results['S0map'] = fullfile(out_dir, "S0map.nii.gz")
              
              if Generate_Residual_Rsquare_Maps:
                  self._results['Residualmap'] = fullfile(out_dir, "Residualmap.nii.gz")
                  self._results['Rsquaremap'] = fullfile(out_dir, "Rsquaremap.nii.gz")
             
              self._results['tedana_output_bold'] = fullfile(out_dir, "desc-optcom_bold.nii.gz")
              self._results['tedana_choice_selected'] = tedana_choice
              
              if tedana_choice == "echo2" :
                  
                  src_file = in_files[1]
                  tar_file = fullfile(out_dir, "echo2.nii.gz")
                  com_str = "cp -p " + src_file + " " + tar_file
                  com_file = fullfile(out_dir, "echo2_command.txt")
                  with open(com_file, 'w') as fp:
                       fp.write(com_str)
                  
                  shutil.copyfile(src_file, tar_file)    
                  self._results['tedana_output_bold'] = tar_file  ## echo2 data ZSW
                 
                             
     #   self._results['tedana_choice_selected'] = tedana_choice
     #   self._results['tedana_choice_model_fitting_desc'] = tedana_choice
        
        return runtime

def output_model_fitting_data(model_fitting_data, output_dir, source_file, output_entities, write_data_method):
     if write_data_method == "Use_DerivativesDataSink":
         from fmriprep.utils.meepi import combine_meepi_source
         
         if Generate_Residual_Rsquare_Maps:
               inputnode = pe.Node(niu.IdentityInterface(fields=['source_file', 'T2starmap', 'S0map', 'Residualmap', 'Rsquaremap']),name='inputnode')
         else:
               inputnode = pe.Node(niu.IdentityInterface(fields=['source_file', 'T2starmap', 'S0map']),name='inputnode')
         
         inputnode.inputs.source_file = combine_meepi_source(source_file)
         inputnode.inputs.T2starmap = model_fitting_data['T2starmap']
         inputnode.inputs.S0map = model_fitting_data['S0map']
         
         if Generate_Residual_Rsquare_Maps:
              inputnode.inputs.Residualmap = model_fitting_data['Residualmap']
              inputnode.inputs.Rsquaremap = model_fitting_data['Rsquaremap']
         
         ds_T2starmap = pe.Node(
             DerivativesDataSink(base_directory=output_dir, suffix='T2starmap',
                            compress=True, dismiss_entities=("echo",)),
             name='ds_T2starmap', run_without_submitting=True,
             mem_gb=DEFAULT_MEMORY_MIN_GB)
    
         ds_S0map = pe.Node(
             DerivativesDataSink(base_directory=output_dir, suffix='S0map',
                            compress=True, dismiss_entities=("echo",)),
             name='ds_S0map', run_without_submitting=True,
             mem_gb=DEFAULT_MEMORY_MIN_GB)
    
         if Generate_Residual_Rsquare_Maps:
              ds_Residualmap = pe.Node(
                     DerivativesDataSink(base_directory=output_dir, suffix='Residualmap',
                            compress=True, dismiss_entities=("echo",)),
                     name='ds_Residualmap', run_without_submitting=True,
                     mem_gb=DEFAULT_MEMORY_MIN_GB)
    
         if Generate_Residual_Rsquare_Maps:
              ds_Rsquaremap = pe.Node(
                     DerivativesDataSink(base_directory=output_dir, suffix='Rsquaremap',
                            compress=True, dismiss_entities=("echo",)),
                            name='ds_Rsquaremap', run_without_submitting=True,
                            mem_gb=DEFAULT_MEMORY_MIN_GB)
   
         workflow = pe.Workflow(name='output_model_fitting_wf')
         workflow.base_dir = output_dir

         if Generate_Residual_Rsquare_Maps:
              workflow.connect([
                  (inputnode, ds_T2starmap, [('source_file', 'source_file'),
                                                ('T2starmap', 'in_file'),
                                               ]),
   
                   (inputnode, ds_S0map, [('source_file', 'source_file'),
                                            ('S0map', 'in_file'),
                                           ]),
   
                   (inputnode, ds_Residualmap, [('source_file', 'source_file'),
                                                  ('Residualmap', 'in_file'),
                                                 ]),
   
                    (inputnode, ds_Rsquaremap, [('source_file', 'source_file'),
                                                 ('Rsquaremap', 'in_file'),
                                                ]),
                    ])
         else:
              workflow.connect([
                  (inputnode, ds_T2starmap, [('source_file', 'source_file'),
                                                ('T2starmap', 'in_file'),
                                               ]),
   
                   (inputnode, ds_S0map, [('source_file', 'source_file'),
                                            ('S0map', 'in_file'),
                                           ]),
   
                    ])
         workflow.run()
     if write_data_method == "Directly_write_to_files":
          if Generate_Residual_Rsquare_Maps:
               model_fitting_keys =  ['T2starmap', 'S0map', 'Residualmap', 'Rsquaremap']
          else:
              model_fitting_keys =  ['T2starmap', 'S0map']
          bids_keys = ['subject', 'session', 'task', 'run']
          subj_prefix = ""
          for b_key in bids_keys:
              if b_key in output_entities:
                  if b_key == "subject":
                      subj_prefix = b_key.replace("subject", "sub") + "-" + output_entities[b_key]
                  else:
                      subj_prefix = subj_prefix + "_" + b_key.replace("session", "ses") + "-" + str(output_entities[b_key])   
          ses_key = "session"
          subject_name = "sub-" + output_entities["subject"]
          if ses_key in output_entities:
              session_name = "ses-" + output_entities[ses_key]
              outDir = fullfile(output_dir, subject_name, session_name, "func")
          else:
              outDir = fullfile(output_dir, subject_name, "func")
          if not os.path.exists(outDir):
              os.makedirs(outDir)
          for m_key in model_fitting_keys:
              src_data_file = model_fitting_data[m_key]
              tar_data_fname = subj_prefix + "_" + m_key + ".nii.gz"
              tar_data_file  = fullfile(outDir, tar_data_fname)
              shutil.copyfile(src_data_file, tar_data_file)
         
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