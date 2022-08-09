import os
import os.path as osp
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
DEFAULT_INPUT_PATH = Path("/input")
DEFAULT_ALGORITHM_OUTPUT_IMAGES_PATH = Path("/output/images/stroke-lesion-segmentation")


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
        prediction_path = os.path.join(inputdir_highres, input_filename + '.nii.gz')
    else:
        prediction_path = os.path.join(inputdir_lowres, input_filename + '.nii.gz')
    
    # dwi_nrray = sitk.GetArrayFromImage(dwi_sitk_img)
    prediction_sitk_img = sitk.ReadImage(prediction_path)

    # prediction_array = sitk.GetArrayFromImage(prediction_sitk_img)
    # dwi_lesion_area = dwi_nrray * (prediction_array == 1)
    # dwi_lesion_area_mask = dwi_lesion_area * (dwi_lesion_area > 127)
    # dwi_lesion_area_mask = (dwi_lesion_area_mask > 0).astype(int)

    # modify_prediction_sitk_img = sitk.GetImageFromArray(dwi_lesion_area_mask)
    # modify_prediction_sitk_img.CopyInformation(prediction_sitk_img)

    # sitk.WriteImage(modify_prediction_sitk_img, os.path.join(outputdir, input_filename + '.mha'))
    sitk.WriteImage(prediction_sitk_img, os.path.join(outputdir, input_filename + '.mha'))

    print("Done")