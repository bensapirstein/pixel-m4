# treebank, directories and hyperparameters
TREEBANK="UD_Japanese-GSD"
DATA_DIR="./data/ud-treebanks-v2.10/${TREEBANK}"
FALLBACK_FONTS_DIR="./fallback_fonts_dir"
SEED=0
LR=1e-5

python ./scripts/training/run_ud_bigrams.py \
	--model_name_or_path="Team-PIXEL/pixel-m4" \
	--remove_unused_columns=False \
	--data_dir=${DATA_DIR} \
	--do_train --do_eval --do_predict \
	--dropout_prob=0.1 \
	--max_seq_length=256 \
	--max_steps=15000 \
	--early_stopping \
	--early_stopping_patience=5 \
	--per_device_train_batch_size=64 \
	--gradient_accumulation_steps=1 \
	--learning_rate=${LR} \
	--warmup_steps=100 \
	--run_name="pixel-m4--debug" \
	--output_dir=./debug/pixel-m4/udp/${TREEBANK}/${LR}--${SEED} \
	--overwrite_output_dir \
	--overwrite_cache \
	--logging_strategy=steps \
	--logging_steps=100 \
	--evaluation_strategy=steps \
	--eval_steps=500 \
	--save_strategy=steps \
	--save_steps=500 \
	--save_total_limit=1 \
	--report_to=none \
	--log_predictions \
	--load_best_model_at_end=True \
	--metric_for_best_model="eval_las" \
	--bf16 \
	--half_precision_backend=cuda_amp \
	--fallback_fonts_dir=${FALLBACK_FONTS_DIR} \
	--seed=0 \
	--dataloader_num_workers=8 \
	--rendering_backend="pangocairo"

    # --save_only_model=True \