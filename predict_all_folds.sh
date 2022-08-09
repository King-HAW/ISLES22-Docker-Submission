# all commands to run for predict
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

NNUNet_FOLDER=$SCRIPTPATH"/nnUNet/nnunet/"
HIGHRES_DATA_FOLDER="/nnunet_data_highres"
LOWRES_DATA_FOLDER="/nnunet_data_lowres"
HIGHRES_TASK_NAME="Task120_ISLES_22"
LOWRES_TASK_NAME="Task121_ISLES_22_lowres"
THREADS=16

if [ `ls $HIGHRES_DATA_FOLDER | wc -l` -gt 0 ]; then

    # high res unet model
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o u_model_h_input_fold0 -tr nnUNetTrainerV2_500epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 0 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o u_model_h_input_fold1 -tr nnUNetTrainerV2_500epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 1 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o u_model_h_input_fold2 -tr nnUNetTrainerV2_500epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 2 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o u_model_h_input_fold3 -tr nnUNetTrainerV2_500epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 3 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o u_model_h_input_fold4 -tr nnUNetTrainerV2_500epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 4 -z

    # high res resunet model
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o r_model_h_input_fold0 -tr nnUNetTrainerV2_800epochs_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 0 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o r_model_h_input_fold1 -tr nnUNetTrainerV2_800epochs_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 1 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o r_model_h_input_fold2 -tr nnUNetTrainerV2_800epochs_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 2 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o r_model_h_input_fold3 -tr nnUNetTrainerV2_800epochs_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 3 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o r_model_h_input_fold4 -tr nnUNetTrainerV2_800epochs_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 4 -z

    # high res unet model bd loss
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o u_model_h_input_bd_loss_fold0 -tr nnUNetTrainerV2_FT_100epochs_Loss_Boundary --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 0 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o u_model_h_input_bd_loss_fold1 -tr nnUNetTrainerV2_FT_100epochs_Loss_Boundary --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 1 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o u_model_h_input_bd_loss_fold2 -tr nnUNetTrainerV2_FT_100epochs_Loss_Boundary --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 2 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o u_model_h_input_bd_loss_fold3 -tr nnUNetTrainerV2_FT_100epochs_Loss_Boundary --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 3 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $HIGHRES_DATA_FOLDER -o u_model_h_input_bd_loss_fold4 -tr nnUNetTrainerV2_FT_100epochs_Loss_Boundary --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $HIGHRES_TASK_NAME -m 3d_fullres -f 4 -z

    # ensemble high res predictions
    python $NNUNet_FOLDER"inference/ensemble_predictions.py" -t $THREADS -o ensemble/highres_predictions -f u_model_h_input_fold0 u_model_h_input_fold1 u_model_h_input_fold2 u_model_h_input_fold3 u_model_h_input_fold4 r_model_h_input_fold0 r_model_h_input_fold1 r_model_h_input_fold2 r_model_h_input_fold3 r_model_h_input_fold4 u_model_h_input_bd_loss_fold0 u_model_h_input_bd_loss_fold1 u_model_h_input_bd_loss_fold2 u_model_h_input_bd_loss_fold3 u_model_h_input_bd_loss_fold4

fi

if [ `ls $LOWRES_DATA_FOLDER | wc -l` -gt 0 ]; then

    # low res unet model
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $LOWRES_DATA_FOLDER -o u_model_l_input_fold0 -tr nnUNetTrainerV2_500epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $LOWRES_TASK_NAME -m 3d_fullres -f 0 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $LOWRES_DATA_FOLDER -o u_model_l_input_fold1 -tr nnUNetTrainerV2_500epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $LOWRES_TASK_NAME -m 3d_fullres -f 1 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $LOWRES_DATA_FOLDER -o u_model_l_input_fold2 -tr nnUNetTrainerV2_500epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $LOWRES_TASK_NAME -m 3d_fullres -f 2 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $LOWRES_DATA_FOLDER -o u_model_l_input_fold3 -tr nnUNetTrainerV2_500epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $LOWRES_TASK_NAME -m 3d_fullres -f 3 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $LOWRES_DATA_FOLDER -o u_model_l_input_fold4 -tr nnUNetTrainerV2_500epochs_Loss_DiceTopK10 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $LOWRES_TASK_NAME -m 3d_fullres -f 4 -z

    # low res resunet model
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $LOWRES_DATA_FOLDER -o r_model_l_input_fold0 -tr nnUNetTrainerV2_500epochs_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $LOWRES_TASK_NAME -m 3d_fullres -f 0 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $LOWRES_DATA_FOLDER -o r_model_l_input_fold1 -tr nnUNetTrainerV2_500epochs_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $LOWRES_TASK_NAME -m 3d_fullres -f 1 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $LOWRES_DATA_FOLDER -o r_model_l_input_fold2 -tr nnUNetTrainerV2_500epochs_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $LOWRES_TASK_NAME -m 3d_fullres -f 2 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $LOWRES_DATA_FOLDER -o r_model_l_input_fold3 -tr nnUNetTrainerV2_500epochs_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $LOWRES_TASK_NAME -m 3d_fullres -f 3 -z
    python $NNUNet_FOLDER"inference/predict_simple.py" -i $LOWRES_DATA_FOLDER -o r_model_l_input_fold4 -tr nnUNetTrainerV2_500epochs_ResencUNet_DA3 -p nnUNetPlans_FabiansResUNet_v2.1 --num_threads_preprocessing $THREADS --num_threads_nifti_save $THREADS -t $LOWRES_TASK_NAME -m 3d_fullres -f 4 -z

    # ensemble low res predictions
    python $NNUNet_FOLDER"inference/ensemble_predictions.py" -t $THREADS -o ensemble/lowres_predictions -f u_model_l_input_fold0 u_model_l_input_fold1 u_model_l_input_fold2 u_model_l_input_fold3 u_model_l_input_fold4 r_model_l_input_fold0 r_model_l_input_fold1 r_model_l_input_fold2 r_model_l_input_fold3 r_model_l_input_fold4

fi