# Sample code for combining audio and visual (ultrasound) features at the
# DNN stage of an HMM-DNN hybrid model.
# The goal is to get the best alignment of AV features with text,
# to be used by speech therapists.

# Previous stages (audio-only HMM-GMM baseline) are not shown until I receive
# permission from the CSTR at the University of Edinburgh. These include:
    # A 5 state (3-emitting) monophone acoustic model (“mono0a”).
    # Further triphone models with more complex feature transformations,
    # including delta and delta-delta features (“tri1”), LDA-MLLT (“tri2b”)
    # and SAT (“tri3b”)

# The following version uses autoencoder features of images,
# the AE trained on 10k images.
# This is the best performing model, which improved the audio-only baseline
# alignment precision/recall with a gold standard alignment.
# Other experiments are with early feature fusion, concatenating AV before
# monophone training which performs poorly (i.e. 9 DCT features + 13 MFCCs).

EXP_DIR=/exp
#1. Make fmllr features for audio, using alignment from audio-only
# triphone training, tri3b

# UXTD = typical speech, UXSSD or UPX  = disordered speech
# nnet data: audio UXTD+UPX datasets combined
NNET_DATA_DIR=${EXP_DIR}/tri4_nnet/nnet_data
gmm_dir=${EXP_DIR}/tri3b
for subset in train cv; do
     steps/nnet/make_fmllr_feats.sh --nj ${sub_nj} --cmd "$train_cmd" \
           --transform-dir ${gmm_dir} \
           ${NNET_DATA_DIR}/fmllr/${subset} \
           ${TRAIN_DATA}/nnet_${subset} ${gmm_dir} \
           ${NNET_DATA_DIR}/fmllr/${subset}/log \
           ${NNET_DATA_DIR}/fmllr/${subset}/data
             # re-align train/cv data
    steps/align_fmllr.sh --nj ${sub_nj} --cmd "$train_cmd" \
            ${TRAIN_DATA}/nnet_${subset} ${LANG_DIR} ${gmm_dir} \
            ${NNET_DATA_DIR}/tri3b_ali/${subset}
done

# 2. concatenate fmllr audio-only features (40-dimensions)
# with autoencoder visual features (128-dimensions), "ae10k".
# (note, other experiments apply fMLLR to AV instead of audio-only)
# fMLLR + AE10k training data fused and put at: data/train/upx_uxtd_train_FMLLR_AE
steps/append_feats.sh exp/tri4_nnet/nnet_data/fmllr/train \
data/train/upx_uxtd_p_AE_train \
data/train/upx_uxtd_train_FMLLR_AE \
exp/tri4_nnet/nnet_data/AE-late \
AE2
# fMLLR + AE10k dev data fused and put at: data/train/upx_uxtd_cv_FMLLR_AE
./steps/append_feats.sh expREDUCEDaudUPXUXTD/tri4_nnet/nnet_data/fmllr/cv \
data/train/upx_uxtd_p_AE_cv \
data/train/upx_uxtd_cv_FMLLR_AE \
expREDUCEDaudUPXUXTD/tri4_nnet/nnet_data/AE-late \
AE3
# 3. compute CMVN for concatenated features
steps/compute_cmvn_stats.sh data/train/upx_uxtd_train_FMLLR_AE
steps/compute_cmvn_stats.sh data/train/upx_uxtd_cv_FMLLR_AE
# 4. Train the DNN optimizing per-frame cross-entropy.
# Training data: 40d fmllr + 128d DCT
NNET_TRAIN_DIR=${EXP_DIR}/tri4_nnet/nnet-AE10k
   (tail --pid=$$ -F ${NNET_TRAIN_DIR}/log/train_nnet.log 2>/dev/null)&
   $cuda_cmd ${NNET_TRAIN_DIR}/log/train_nnet.log steps/nnet/train.sh \
       --hid-layers 4 \
       --learn-rate 0.008 \
       data/train/upx_uxtd_train_FMLLR_AE \
       data/train/upx_uxtd_cv_FMLLR_AE \
       ${LANG_DIR} \
       ${NNET_DATA_DIR}/tri3b_ali/train ${NNET_DATA_DIR}/tri3b_ali/cv \
       ${NNET_TRAIN_DIR}
# the DNN is at: exp/tri4_nnet/nnet-AE10k
# Next, must re-align typical and disordered speech with this DNN,
# and evaluate the alignment.
# Shown below is just for disordered speech.
# 5. make fMLLR features for disordered speech audio
steps/nnet/make_fmllr_feats.sh --cmd "$train_cmd" \
      --transform-dir ${gmm_dir}_ali/uxssd/${subset} \
      ${NNET_DATA_DIR}/fmllr/uxssd/${subset} ${uxssd_data}/${subset} ${gmm_dir} \
      ${NNET_DATA_DIR}/fmllr/uxssd/${subset}/log ${NNET_DATA_DIR}/fmllr/uxssd/${subset}/data || exit 1
# 6. combine audio-visual disordered features: fMLLR + AE10k
      for subset in BL Maint Mid Post Therapy; do
        ./steps/append_feats.sh ${EXP_DIR}/tri4_nnet/nnet_data/fmllr/uxssd/${subset} \
        data/data_uxssd_ae10k/${subset} \
        data/uxssd_AV_fmllr_ae10k/${subset} exp/uxssd-av-ae10k ae
            done
# 7. CMVN for new AV data
      for subset in BL Maint Mid Post Therapy; do
        steps/compute_cmvn_stats.sh data/uxssd_AV_fmllr_ae10k/${subset}
            done
# 8. Align the disordered data
for subset in BL Maint Mid Post Therapy; do
    steps/nnet/align.sh \
    data/uxssd_AV_fmllr_ae10k/${subset} ${LANG_DIR} ${NNET_TRAIN_DIR} ${EXP_DIR}/tri4_nnet_ali/uxssd_AE10k/${subset}
done
# 9. Score the alignment by comparing it by the second to a gold standard alignment,
#  hand labeled by a speech-language therapist
for subset in BL Maint Mid Post Therapy; do
    local/align/ali-to-word.sh data/uxssd_AV_ae10k/${subset} \
    ${LANG_DIR} ${EXP_DIR}/tri4_nnet_ali/uxssd_AE10k/${subset} \
    local/align/reference/uxssd/${subset}
done
