CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
    --main_process_port 22003 \
    main.py \
    --model=Pctx \
    --dataset=AmazonReviews2023 \
    --category=Video_Games \
    --run_GR_or_not=True \
    --rq_faiss=True \
    --n_inference_ensemble=5 \
    --augmentation_probability=0.6 \
    --test_ckpt='game_pctx.pth' \
    --run_mode='test'
