import os
import os.path as osp
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
DEFAULT_INPUT_PATH = Path("/input")


def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkBSpline):
    target_Size = target_img.GetSize()
    target_Spacing = target_img.GetSpacing()
    target_origin = target_img.GetOrigin()
    target_direction = target_img.GetDirection()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)
    resampler.SetSize(target_Size)
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)
    return itk_img_resampled


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
    outputdir_highres = Path('/nnunet_data_highres')
    outputdir_lowres = Path('/nnunet_data_lowres')

    dwi_image_path = str(get_file_path(input_path, slug='dwi-brain-mri', filetype='image'))
    adc_image_path = str(get_file_path(input_path, slug='adc-brain-mri', filetype='image'))
    flair_image_path = str(get_file_path(input_path, slug='flair-brain-mri', filetype='image'))

    dwi_sitk_img = sitk.ReadImage(dwi_image_path)
    adc_sitk_img = sitk.ReadImage(adc_image_path)
    flair_sitk_img = sitk.ReadImage(flair_image_path)
    dwi_spcing = dwi_sitk_img.GetSpacing()
    input_filename = dwi_image_path.split('/')[-1].replace('.mha', '')

    # write three files (adc:0000, dwi:0001, 'flair':0002)
    # check whether flair has the same shape:
    if np.max(dwi_spcing) < 3.0: # highres image
        if not flair_sitk_img.GetSize() == dwi_sitk_img.GetSize():
            flair_resample_image = resize_image_itk(flair_sitk_img, dwi_sitk_img)
            sitk.WriteImage(flair_resample_image, osp.join(outputdir_highres, '{}_0002.nii.gz'.format(input_filename)))
        else:
            sitk.WriteImage(flair_sitk_img, osp.join(outputdir_highres, '{}_0002.nii.gz'.format(input_filename)))

        sitk.WriteImage(adc_sitk_img, osp.join(outputdir_highres, '{}_0000.nii.gz'.format(input_filename)))
        sitk.WriteImage(dwi_sitk_img, osp.join(outputdir_highres, '{}_0001.nii.gz'.format(input_filename)))

    else:
        if not flair_sitk_img.GetSize() == dwi_sitk_img.GetSize():
            flair_resample_image = resize_image_itk(flair_sitk_img, dwi_sitk_img)
            sitk.WriteImage(flair_resample_image, osp.join(outputdir_lowres, '{}_0002.nii.gz'.format(input_filename)))
        else:
            sitk.WriteImage(flair_sitk_img, osp.join(outputdir_lowres, '{}_0002.nii.gz'.format(input_filename)))

        sitk.WriteImage(adc_sitk_img, osp.join(outputdir_lowres, '{}_0000.nii.gz'.format(input_filename)))
        sitk.WriteImage(dwi_sitk_img, osp.join(outputdir_lowres, '{}_0001.nii.gz'.format(input_filename)))

    print("Done")