CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --model=Pctx \
    --dataset=AmazonReviews2023 \
    --category=Musical_Instruments \
    --run_GR_or_not=False \
    --refresh_cluster_result=True \
    --pretrained_model_path=Duorec_pretrained_Emb \
    --pretrained_model_name=instrument_duorec.pth \
    --n_groups=10 \
    --distance=4 \
    --start=2 \
    --k_gamma=4.2