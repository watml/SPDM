# Default model paramters
BS=64

DATASET_NAME=$1
PRED=$2

NGPU=1

SIGMA_MAX=80.0
SIGMA_MIN=0.002
SIGMA_DATA=0.5
COV_XY=0
NUM_CH=256
NUM_HEADS=64
ATTN=32,16,8
SAMPLER=real-uniform
NUM_RES_BLOCKS=2
USE_16FP=True
ATTN_TYPE=reg
TEST_INTERVAL=5000
IN_CHANNELS=3
OUT_CHANNELS=3

# Arguments for group equivarient layers
G_EQUIV=False
G_OUTPUT="C4_S"
# Argument for group regularization
G_REG=False

# Arguments specific to each dataset
if [[ $DATASET_NAME == "e2h" ]]; then
    DATA_DIR=YOUR_DATASET_PATH
    DATASET=edges2handbags
    IMG_SIZE=64
    NUM_CH=192
    NUM_RES_BLOCKS=3
    EXP="e2h${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=100000
elif [[ "$DATASET_NAME" == "lysto64" ]]; then
    DATA_DIR="/home/datasets/lysto64_random_crop_ddbm"
    DATASET="lysto64"
    IMG_SIZE=64
    NUM_CH=128
    NUM_RES_BLOCKS=3
    EXP="lysto64${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=5000
elif [[ "$DATASET_NAME" == "ct_pet" ]]; then
    DATA_DIR="/data/ct_pet"
    DATASET="ct_pet"
    IMG_SIZE=256
    SIGMA_MAX=20.0
    SIGMA_MIN=0.0005
    INV_REG="H"
    EXP="ct_pet${IMG_SIZE}_${NUM_CH}d_INV_${INV_REG}"
    SAVE_ITER=20000
    AUTOENCODER_CKPT=/logs/pet_ct_ssim_lpips_l2_lr1e-5/checkpoint_700000.pth
fi
    
# Arguments for each of the possible diffusion noise schedules
if  [[ $PRED == "ve" ]]; then
    EXP+="_ve"
    COND=concat
elif  [[ $PRED == "vp" ]]; then
    EXP+="_vp"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
elif  [[ $PRED == "ve_simple" ]]; then
    EXP+="_ve_simple"
    COND=concat
elif  [[ $PRED == "vp_simple" ]]; then
    EXP+="_vp_simple"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
else
    echo "Not supported"
    exit 1
fi

# Arguments for image size in relation to batch size for training
if  [[ $IMG_SIZE == 256 ]]; then
    BS=16
elif  [[ $IMG_SIZE == 128 ]]; then
    BS=16
elif  [[ $IMG_SIZE == 128 ]]; then
    BS=14
elif  [[ $IMG_SIZE == 64 ]]; then
    BS=30
else
    echo "Not supported"
    exit 1
fi