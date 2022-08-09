FROM nvcr.io/nvidia/pytorch:22.01-py3



RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output /nnunet_data_highres /nnunet_data_lowres \
    && chown algorithm:algorithm /opt/algorithm /input /output /nnunet_data_highres /nnunet_data_lowres

USER algorithm

WORKDIR /opt/algorithm

# ENV HOME="/home/algorithm/.local/bin"
ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm nnUNet /opt/algorithm/nnUNet
RUN cd /opt/algorithm/nnUNet \
	&& pip install --no-cache-dir -e .

COPY --chown=algorithm:algorithm nnUNet_trained_models /opt/algorithm/nnUNet_trained_models
COPY --chown=algorithm:algorithm run_segmentation.py /opt/algorithm/
COPY --chown=algorithm:algorithm predict.sh /opt/algorithm/
COPY --chown=algorithm:algorithm predict_all_folds.sh /opt/algorithm/
COPY --chown=algorithm:algorithm copy_images_to_nnunet_format.py /opt/algorithm/
COPY --chown=algorithm:algorithm rename_predictions_to_mha.py /opt/algorithm/
COPY --chown=algorithm:algorithm process.py /opt/algorithm/

ENTRYPOINT python -m run_segmentation $0 $@
