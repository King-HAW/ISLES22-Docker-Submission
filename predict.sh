#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

export nnUNet_raw_data_base=$SCRIPTPATH"/nnUNet_raw/"
export nnUNet_preprocessed=$SCRIPTPATH"/nnUNet_preprocessed/"
export RESULTS_FOLDER=$SCRIPTPATH"/nnUNet_trained_models/"

python $SCRIPTPATH/copy_images_to_nnunet_format.py --debug_flag False

mkdir -p u_model_h_input_fold0 u_model_h_input_fold1 u_model_h_input_fold2 u_model_h_input_fold3 u_model_h_input_fold4
mkdir -p u_model_h_input_bd_loss_fold0 u_model_h_input_bd_loss_fold1 u_model_h_input_bd_loss_fold2 u_model_h_input_bd_loss_fold3 u_model_h_input_bd_loss_fold4
mkdir -p u_model_l_input_fold0 u_model_l_input_fold1 u_model_l_input_fold2 u_model_l_input_fold3 u_model_l_input_fold4
mkdir -p r_model_h_input_fold0 r_model_h_input_fold1 r_model_h_input_fold2 r_model_h_input_fold3 r_model_h_input_fold4
mkdir -p r_model_l_input_fold0 r_model_l_input_fold1 r_model_l_input_fold2 r_model_l_input_fold3 r_model_l_input_fold4
mkdir -p ensemble/highres_predictions ensemble/lowres_predictions

bash $SCRIPTPATH/predict_all_folds.sh

python $SCRIPTPATH/rename_predictions_to_mha.py --debug_flag False

python -m process $0 $@

rm -rf u_model_h_input_fold0 u_model_h_input_fold1 u_model_h_input_fold2 u_model_h_input_fold3 u_model_h_input_fold4
rm -rf u_model_h_input_bd_loss_fold0 u_model_h_input_bd_loss_fold1 u_model_h_input_bd_loss_fold2 u_model_h_input_bd_loss_fold3 u_model_h_input_bd_loss_fold4
rm -rf u_model_l_input_fold0 u_model_l_input_fold1 u_model_l_input_fold2 u_model_l_input_fold3 u_model_l_input_fold4
rm -rf r_model_h_input_fold0 r_model_h_input_fold1 r_model_h_input_fold2 r_model_h_input_fold3 r_model_h_input_fold4
rm -rf r_model_l_input_fold0 r_model_l_input_fold1 r_model_l_input_fold2 r_model_l_input_fold3 r_model_l_input_fold4

