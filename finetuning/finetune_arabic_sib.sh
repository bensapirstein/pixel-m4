export WANDB_DISABLED=true
export FONTCONFIG_PATH=/etc/fonts  # some systems needs this.

MODEL="Team-PIXEL/pixel-m4"
LANG="arz_Arab"

# fixed hyperparameters for SIB-200 experiments.
FALLBACK_FONTS_DIR="./fallback_fonts"  # let's say this is where we downloaded the fonts to
SEQ_LEN=256
BSZ=32
GRAD_ACCUM=1
NUM_STEPS=15000
FP16_OR_BF16_="bf16"

# sweeped hyperparameters.
SEED=0
LR=1e-5

# set this based on your compute.
NUM_WORKERS=8

RUN_NAME="${LANG}-$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"
OUTPUT_DIR="./logs/$(basename ${MODEL})/sib-200/${LANG}/${LR}--${SEED}-7"
DATA_DIR="./data/sib-200"
python ./scripts/training/run_sib_bigrams.py \
    --model_name_or_path=${MODEL} \
    --remove_unused_columns=False \
    --data_dir=${DATA_DIR} \
    --language ${LANG} \
    --do_train --do_eval --do_predict \
    --dropout_prob=0.1 \
    --max_seq_length=${SEQ_LEN} \
    --max_steps=${NUM_STEPS} \
    --early_stopping=False\
    --early_stopping_patience=20 \
    --per_device_train_batch_size=${BSZ} \
    --gradient_accumulation_steps=${GRAD_ACCUM} \
    --learning_rate=${LR} \
    --run_name=${RUN_NAME} \
    --output_dir=${OUTPUT_DIR} \
    --overwrite_output_dir \
    --overwrite_cache \
    --logging_strategy=epoch \
    --logging_steps=1 \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --save_total_limit=2 \
    --report_to=none \
    --load_best_model_at_end=True \
    --metric_for_best_model="eval_f1" \
    --bf16 \
    --half_precision_backend=cuda_amp \
    --fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
    --seed=${SEED} \
    --dataloader_num_workers=${NUM_WORKERS} \
    --rendering_backend="bigrams"
    # --log_predictions \