BASE_DIR="D:/NAR/NAR-Former-V2-main/NAR-Former-V2-main"
DATASET_DIR="$BASE_DIR/dataset/unseen_structure"

python $BASE_DIR/predictor/main.py \
    --only_test \
    --gpu 0 \
    --batch_size 1 \
    --data_root "$DATASET_DIR/data" \
    --all_latency_file "${DATASET_DIR}/help.txt" \
    --norm_sf \
    --onnx_dir "${DATASET_DIR}" \
    --log "log/test.log" \
    --pretrain "checkpoints/ckpt_best.pth" \
    --ckpt_save_freq 1000 \
    --test_freq 1 \
    --print_freq 50 \
    --embed_type trans \
    --num_node_features 152 \
    --glt_norm LN \
    --train_test_stage \
    --use_degree \