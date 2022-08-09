import os
import os.path as osp
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import *
from typing import Union, Tuple
from nnunet.preprocessing.preprocessing import get_lowres_axis, get_do_separate_z, resample_data_or_seg

DEFAULT_INPUT_PATH = Path("/input")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("/output/images/stroke-lesion-segmentation")


def save_segmentation_nifti_from_softmax(segmentation_softmax: Union[str, np.ndarray], out_fname: str,
                                         properties_dict: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):

    if verbose: print("force_separate_z:", force_separate_z, "interpolation order:", order)

    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if lowres_axis is not None and len(lowres_axis) != 1:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False

        if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z,
                                               order_z=interpolation_order_z)
        # seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    else:
        if verbose: print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    if region_class_order is None:
        fg_softmax = seg_old_spacing[1]
        fg_ones_thres_1 = np.zeros(fg_softmax.shape)
        fg_ones_thres_2 = np.zeros(fg_softmax.shape)
        fg_ones_thres_1[fg_softmax > 0.5] = 1
        fg_ones_thres_2[fg_softmax > 0.4] = 1

        if np.sum(fg_ones_thres_1) == 0:
            tmp_thresh = np.max(fg_softmax) - 0.05
            fg_ones_thres_tmp = np.zeros(fg_softmax.shape)
            fg_ones_thres_tmp[fg_softmax > tmp_thresh] = 1
            seg_old_spacing = fg_ones_thres_tmp

        else:
            seg_old_spacing = fg_ones_thres_1

    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
        seg_old_spacing = seg_old_spacing_final

    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping, dtype=np.uint8)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)


def get_file_path(input_path, slug, filetype='image'):
        """ Gets the path for each MR image/json file."""

        if filetype == 'image':
            file_list = list((input_path / "images" / slug).glob("*.mha"))
        elif filetype == 'json':
            file_list = list(input_path.glob("*{}.json".format(slug)))

        # Check that there is a single file to load.
        if len(file_list) != 1:
            print('Loading error')
        else:
            return file_list[0]


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_flag', type=bool, default=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = getargs()

    debug_flag = args.debug_flag  # False for running the docker!

    input_path = DEFAULT_INPUT_PATH
    inputdir_highres= Path('/opt/algorithm/ensemble/highres_predictions')
    inputdir_lowres= Path('/opt/algorithm/ensemble/lowres_predictions')
    outputdir = DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    dwi_image_path = str(get_file_path(input_path, slug='dwi-brain-mri', filetype='image'))
    dwi_sitk_img = sitk.ReadImage(dwi_image_path)
    dwi_spcing = dwi_sitk_img.GetSpacing()
    input_filename = dwi_image_path.split('/')[-1].replace('.mha', '')

    # write three files (adc:0000, dwi:0001, 'flair':0002)
    if np.max(dwi_spcing) < 3.0: # highres image
        prediction_nii_path = os.path.join(inputdir_highres, input_filename + '.nii.gz')
        prediction_pkl_path = os.path.join(inputdir_highres, input_filename + '.pkl')
        prediction_npz_path = os.path.join(inputdir_highres, input_filename + '.npz')
    else:
        prediction_nii_path = os.path.join(inputdir_lowres, input_filename + '.nii.gz')
        prediction_pkl_path = os.path.join(inputdir_lowres, input_filename + '.pkl')
        prediction_npz_path = os.path.join(inputdir_lowres, input_filename + '.npz')
    
    # dwi_nrray = sitk.GetArrayFromImage(dwi_sitk_img)

    ### Save mha file
    # prediction_sitk_img = sitk.ReadImage(prediction_nii_path)
    # sitk.WriteImage(prediction_sitk_img, os.path.join(outputdir, input_filename + '.mha'))

    ### Postprocessing Part
    softmax = np.load(prediction_npz_path)['softmax']
    props = load_pickle(prediction_pkl_path)
    out_file = os.path.join(outputdir, input_filename + '.mha')

    regions_class_order = None

    save_segmentation_nifti_from_softmax(softmax, out_file, props[0], 3, regions_class_order, None, None, force_separate_z=None)

    # sitk.WriteImage(prediction_sitk_img, os.path.join(outputdir, input_filename + '.mha'))

    print("Done")