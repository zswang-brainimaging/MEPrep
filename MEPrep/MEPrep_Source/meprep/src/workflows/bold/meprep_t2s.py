# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Generate T2* map from multi-echo BOLD images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_t2s_wf

"""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from fmriprep.interfaces.maths import Label2Mask, Clip
from fmriprep.interfaces.reports import LabeledHistogram

from fmriprep.interfaces.meprep_multiecho import run_tedana  ##ZSW

from fmriprep import config   ##ZSW

from fmriprep.interfaces.preICA_se_interface import preICA   ##ZSW
from fmriprep.interfaces.preICA_apply_mask import apply_mask    ##ZSW
from fmriprep.interfaces.preICA_register import register   ###ZSW

#from fmriprep.interfaces.preICA_tensor_interface import run_TensorICA ##ZSW
from fmriprep.interfaces.preICA_tensor_interface import run_TensorICA ##ZSW
from fmriprep.interfaces.preICA_tcme_interface import run_tcmeICA ##ZSW
from fmriprep.interfaces.preICA_scme_interface import run_scmeICA  ##ZSW

Use_Tensor_ICA = False
Use_tcme_ICA = False
Use_scme_ICA = True

LOGGER = config.loggers.workflow

write_wf_to_graph = config.execution.write_graph


# pylint: disable=R0914
def init_bold_t2s_wf(echo_times, TR, bold_source_file, mem_gb, omp_nthreads,
                     name='bold_t2s_wf'):
    r"""
    Combine multiple echos of :abbr:`ME-EPI (multi-echo echo-planar imaging)`.

    This workflow wraps the `tedana`_ `T2* workflow`_ to optimally
    combine multiple preprocessed echos and derive a T2\ :sup:`★` map.
    The following steps are performed:
    #. Compute the T2\ :sup:`★` map
    #. Create an optimally combined ME-EPI time series

    .. _tedana: https://github.com/me-ica/tedana
    .. _`T2* workflow`: https://tedana.readthedocs.io/en/latest/generated/tedana.workflows.t2smap_workflow.html#tedana.workflows.t2smap_workflow  # noqa

    Parameters
    ----------
    echo_times : :obj:`list` or :obj:`tuple`
        list of TEs associated with each echo
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``bold_t2s_wf``)

    Inputs
    ------
    bold_file
        list of individual echo files
    bold_mask
        a binary mask to apply to the BOLD files

    Outputs
    -------
    bold
        the optimally combined time series for all supplied echos
    t2star_map
        the calculated T2\ :sup:`★` map

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A T2<sup>★</sup> map was estimated from the preprocessed EPI echoes, by voxel-wise fitting
the maximal number of echoes with reliable signal in that voxel to a monoexponential signal
decay model with nonlinear regression.
The T2<sup>★</sup>/S<sub>0</sub> estimates from a log-linear regression fit were used for
initial values.
The calculated T2<sup>★</sup> map was then used to optimally combine preprocessed BOLD across
echoes following the method described in [@posse_t2s].
The optimally combined time series was carried forward as the *preprocessed BOLD*.
"""
    def get_first_file(in_files):
        return in_files[0]
    
    tedana_choice_from_config = config.execution.tedana_choice
    preDenoising_choice_from_config = config.execution.preDenoising_choice
    output_echo2_choice_from_config = config.execution.output_echo2
    
    tedana_choice_list = ['optcom', 'optcomDenoised']
    
    
    tedana_choice_with_echo2_dict = {"optcom":["optcom", "echo2"], 
                                     "optcomDenoised":["optcomDenoised", "echo2"], 
                                     "Both":["optcom", "optcomDenoised", "echo2"]}
    
    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file', 'bold_mask', 'movpar_file','source_file']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['bold', 
                                                       't2star_map', 'S0map', # 'Residualmap', 'Rsquaremap', 
                                                       'tedana_choice_selected', 
                                                  #     'tedana_choice_model_fitting_desc'
                                                       ]), 
                         name='outputnode')
    
    apply_mask_node = pe.MapNode(apply_mask(), iterfield=["in_file"], name = "apply_mask_node")
    
    bold2mni_node = pe.Node(register(ref_file='.', out_file='.'), name = "bold2mni_node")
    
    if preDenoising_choice_from_config == "preICA_tensor": # Use_Tensor_ICA:
         preICA_node = pe.Node(run_TensorICA(denoising_type='nonaggr', TR=float(TR)), name='preICAtensor_node')
    elif preDenoising_choice_from_config =="preICA_scme":  #Use_scme_ICA:
         preICA_node = pe.Node(run_scmeICA(denoising_type='nonaggr', TR=float(TR)), name='preICAscme_node')
    elif preDenoising_choice_from_config =="preICA_tcme":  #Use_tcme_ICA:
         preICA_node = pe.Node(run_tcmeICA(denoising_type='nonaggr', TR=float(TR)), name='preICAtcme_node')
    else:
         preICA_node = pe.MapNode(preICA(denoising_type='nonaggr', TR=float(TR)), iterfield=["in_file"], name='preICAse_node')
         

    LOGGER.log(25, 'Generating T2* map and optimally combined ME-EPI time series.')

    node_name_dict = {"optcom":"optcom_node", "optcomDenoised":"optcomDenoised_node", "Both":"tedana_node"}
    node_name_dict_echo2 = {"optcom":"optcom_echo2_node", "optcomDenoised":"optcomDenoised_echo2_node", "Both":"tedana_echo2_node"}
    
    if output_echo2_choice_from_config:
         tedana_node = pe.Node(run_tedana(echo_times=list(echo_times), source_file = list(bold_source_file)), 
                              name=node_name_dict_echo2[tedana_choice_from_config])
         tedana_node.iterables = ("tedana_choice", tedana_choice_with_echo2_dict[tedana_choice_from_config])
    else:    
        if  tedana_choice_from_config =='Both':
            tedana_node = pe.Node(run_tedana(echo_times=list(echo_times), source_file = list(bold_source_file)),
                             name=node_name_dict[tedana_choice_from_config])
            tedana_node.iterables = ("tedana_choice", tedana_choice_list)
        else:
            tedana_node = pe.Node(run_tedana(echo_times=list(echo_times), source_file = list(bold_source_file), 
                                             tedana_choice=tedana_choice_from_config), 
                                                            name=node_name_dict[tedana_choice_from_config])

    LOGGER.log(25, 'Generating T2* map and optimally combined ME-EPI time series.')

    if 'preICA' in preDenoising_choice_from_config:  #=='preICA':
          LOGGER.log(25, 'Calling preICA workflow to denoise fMRIPrep-processed ME-EPI time series before calling TEDANA ...')
     
          workflow.connect([
                           (inputnode, apply_mask_node,[("bold_file", "in_file"),
                                                        ("bold_mask","mask_file")]), 
                           
                           (apply_mask_node, bold2mni_node, [(("masked_bold",get_first_file), "in_file")]),
                           
                           (bold2mni_node, preICA_node, [("bold2mni_matrix_file", "bold2mni_matrix_file")]),
                           
                           (apply_mask_node, preICA_node,[("masked_bold", "in_file"),
                                                #        ("mask_file","mask_file"),
                                                         ]),
                           (inputnode,preICA_node, [("bold_mask", "mask_file"),
                                                     ("movpar_file", "motion_file"),
                                                  #   ("TR","TR"),
                                                  ]),
                           
                           (preICA_node, tedana_node, [("denoised_func_files", "in_files"),
                                                 #    (("bold_mask_file", get_first_file),"mask_file"),
                                                     ]),
                           (inputnode, tedana_node, [("bold_mask", "mask_file")]),
            
                           (tedana_node, outputnode, [('tedana_output_bold', 'bold'),
                                                      ('T2starmap', 't2star_map'),
                                                      ('S0map', 'S0map'),
                                                 #     ('Residualmap', 'Residualmap'),
                                                 #     ('Rsquaremap', 'Rsquaremap'),
                                                      ('tedana_choice_selected', 'tedana_choice_selected'),
                                                 #     ('tedana_choice_model_fitting_desc', 'tedana_choice_model_fitting_desc'),
                                                      ]),
              ])
    else:
          LOGGER.log(25, 'calling TEDANA directly on fMRIPrep-processed ME-EPI time series ...')
          workflow.connect([
                            (inputnode, apply_mask_node,[("bold_file", "in_file"),
                                                         ("bold_mask","mask_file")]), 
                            
                            (apply_mask_node, tedana_node, [("masked_bold", "in_files"),
                                                      ]),
                            (inputnode, tedana_node,  [("bold_mask", "mask_file"),
                                                      ]), 
                            
             
                            (tedana_node, outputnode, [('tedana_output_bold', 'bold'),
                                                       ('T2starmap', 't2star_map'),
                                                       ('S0map', 'S0map'),
                                              #         ('Residualmap', 'Residualmap'),
                                              #         ('Rsquaremap', 'Rsquaremap'),
                                                       ('tedana_choice_selected', 'tedana_choice_selected'),
                                                 #      ('tedana_choice_model_fitting_desc', 'tedana_choice_model_fitting_desc'),
                                                       ]),
              ])

    if write_wf_to_graph:
        # Create color output graph
              
        workflow.write_graph(graph2use='colored', format='png', simple_form=True)
        workflow.write_graph(graph2use="colored", format="svg", simple_form=True)
  
          # Create the detailed graph
        workflow.write_graph(graph2use='flat', format='png', simple_form=True)
        
             
      
        
    return workflow

### Added from t2s v2321 ZSW
def init_t2s_reporting_wf(name: str = 't2s_reporting_wf'):
    r"""
    Generate T2\*-map reports.

    This workflow generates a histogram of estimated T2\* values (in seconds) in the
    cortical and subcortical gray matter mask.

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``t2s_reporting_wf``)

    Inputs
    ------
    t2star_file
        estimated T2\* map
    boldref
        reference BOLD file
    label_file
        an integer label file identifying gray matter with value ``1``
    boldref2anat_xfm
        Affine matrix that maps images in the native bold space into the
        anatomical space of ``label_file``; can be ``"identity"`` if label
        file is already aligned

    Outputs
    -------
    t2star_hist
        an SVG histogram showing estimated T2\* values in gray matter
    t2s_comp_report
        a before/after figure comparing the reference BOLD image and T2\* map
    """
    from nipype.pipeline import engine as pe
    from nireports.interfaces.reporting.base import (
        SimpleBeforeAfterRPT as SimpleBeforeAfter,
    )
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t2star_file', 'boldref', 'label_file', 'boldref2anat_xfm']),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['t2star_hist', 't2s_comp_report']), name='outputnode'
    )

    label_tfm = pe.Node(
        ApplyTransforms(interpolation="MultiLabel", invert_transform_flags=[True]),
        name="label_tfm",
    )

    gm_mask = pe.Node(Label2Mask(label_val=1), name="gm_mask")

    clip_t2star = pe.Node(Clip(maximum=0.1), name="clip_t2star")

    t2s_hist = pe.Node(
        LabeledHistogram(mapping={1: "Gray matter"}, xlabel='T2* (s)'), name='t2s_hist'
    )

    t2s_comparison = pe.Node(
        SimpleBeforeAfter(
            before_label="BOLD Reference",
            after_label="T2* Map",
            dismiss_affine=True,
        ),
        name="t2s_comparison",
        mem_gb=0.1,
    )
    workflow.connect([
        (inputnode, label_tfm, [('label_file', 'input_image'),
                                ('t2star_file', 'reference_image'),
                                ('boldref2anat_xfm', 'transforms')]),
        (inputnode, clip_t2star, [('t2star_file', 'in_file')]),
        (clip_t2star, t2s_hist, [('out_file', 'in_file')]),
        (label_tfm, gm_mask, [('output_image', 'in_file')]),
        (gm_mask, t2s_hist, [('out_file', 'label_file')]),
        (inputnode, t2s_comparison, [('boldref', 'before'),
                                     ('t2star_file', 'after')]),
        (gm_mask, t2s_comparison, [('out_file', 'wm_seg')]),
        (t2s_hist, outputnode, [('out_report', 't2star_hist')]),
        (t2s_comparison, outputnode, [('out_report', 't2s_comp_report')]),
    ])  # fmt:skip
    return workflow

### Added from t2s v22 ZSW
def init_t2s_reporting_wf_v22(name='t2s_reporting_wf'):
    r"""
    Generate T2\*-map reports.

    This workflow generates a histogram of esimated T2\* values (in seconds) in the
    cortical and subcortical gray matter mask.

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of BOLD file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``t2s_reporting_wf``)

    Inputs
    ------
    t2star_file
        estimated T2\* map
    boldref
        reference BOLD file
    label_file
        an integer label file identifying gray matter with value ``1``
    label_bold_xform
        Affine matrix that maps the label file into alignment with the native
        BOLD space; can be ``"identity"`` if label file is already aligned

    Outputs
    -------
    t2star_hist
        an SVG histogram showing estimated T2\* values in gray matter
    t2s_comp_report
        a before/after figure comparing the reference BOLD image and T2\* map
    """
    from nipype.pipeline import engine as pe
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
    from niworkflows.interfaces.reportlets.registration import (
        SimpleBeforeAfterRPT as SimpleBeforeAfter,
    )

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['t2star_file', 'boldref', 'label_file', 'label_bold_xform']
        ),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['t2star_hist', 't2s_comp_report']),
        name='outputnode')

    label_tfm = pe.Node(ApplyTransforms(interpolation="MultiLabel"), name="label_tfm")

    gm_mask = pe.Node(Label2Mask(label_val=1), name="gm_mask")

    clip_t2star = pe.Node(Clip(maximum=0.1), name="clip_t2star")

    t2s_hist = pe.Node(LabeledHistogram(mapping={1: "Gray matter"}, xlabel='T2* (s)'),
                       name='t2s_hist')

    t2s_comparison = pe.Node(
        SimpleBeforeAfter(
            before_label="BOLD Reference",
            after_label="T2* Map",
            dismiss_affine=True,
        ),
        name="t2s_comparison",
        mem_gb=0.1,
    )

    workflow.connect([
        (inputnode, label_tfm, [('label_file', 'input_image'),
                                ('t2star_file', 'reference_image'),
                                ('label_bold_xform', 'transforms')]),
        (inputnode, clip_t2star, [('t2star_file', 'in_file')]),
        (clip_t2star, t2s_hist, [('out_file', 'in_file')]),
        (label_tfm, gm_mask, [('output_image', 'in_file')]),
        (gm_mask, t2s_hist, [('out_file', 'label_file')]),
        (inputnode, t2s_comparison, [('boldref', 'before'),
                                     ('t2star_file', 'after')]),
        (gm_mask, t2s_comparison, [('out_file', 'wm_seg')]),
        (t2s_hist, outputnode, [('out_report', 't2star_hist')]),
        (t2s_comparison, outputnode, [('out_report', 't2s_comp_report')]),
    ])

    return workflow
