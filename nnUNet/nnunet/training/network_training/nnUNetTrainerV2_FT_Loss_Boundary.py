#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.boundary_loss import DC_and_BD_loss


class nnUNetTrainerV2_FT_100epochs_Loss_Boundary(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)        
        # We use the pretrained weight to do the initialization:
        # /data/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task120_ISLES_22/nnUNetTrainerV2_500epochs_Loss_DiceTopK10__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model
        # /data/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task120_ISLES_22/nnUNetTrainerV2_500epochs_Loss_DiceTopK10__nnUNetPlansv2.1/fold_1/model_final_checkpoint.model
        # /data/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task120_ISLES_22/nnUNetTrainerV2_500epochs_Loss_DiceTopK10__nnUNetPlansv2.1/fold_2/model_final_checkpoint.model
        # /data/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task120_ISLES_22/nnUNetTrainerV2_500epochs_Loss_DiceTopK10__nnUNetPlansv2.1/fold_3/model_final_checkpoint.model
        # /data/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task120_ISLES_22/nnUNetTrainerV2_500epochs_Loss_DiceTopK10__nnUNetPlansv2.1/fold_4/model_final_checkpoint.model

        self.initial_lr = 1e-3
        self.max_num_epochs = 100
        self.loss = DC_and_BD_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})


class nnUNetTrainerV2_FT_150epochs_Loss_Boundary(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)        
        # We use the pretrained weight to do the initialization:
        # /data/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task120_ISLES_22/nnUNetTrainerV2_500epochs_Loss_DiceTopK10__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model
        # /data/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task120_ISLES_22/nnUNetTrainerV2_500epochs_Loss_DiceTopK10__nnUNetPlansv2.1/fold_1/model_final_checkpoint.model
        # /data/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task120_ISLES_22/nnUNetTrainerV2_500epochs_Loss_DiceTopK10__nnUNetPlansv2.1/fold_2/model_final_checkpoint.model
        # /data/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task120_ISLES_22/nnUNetTrainerV2_500epochs_Loss_DiceTopK10__nnUNetPlansv2.1/fold_3/model_final_checkpoint.model
        # /data/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task120_ISLES_22/nnUNetTrainerV2_500epochs_Loss_DiceTopK10__nnUNetPlansv2.1/fold_4/model_final_checkpoint.model

        self.initial_lr = 1e-3
        self.max_num_epochs = 150
        self.loss = DC_and_BD_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

