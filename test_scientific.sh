CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
    --main_process_port 22001 \
    main.py \
    --model=Pctx \
    --dataset=AmazonReviews2023 \
    --category=Industrial_and_Scientific \
    --run_GR_or_not=True \
    --rq_faiss=True \
    --n_inference_ensemble=5 \
    --augmentation_probability=0.9 \
    --test_ckpt='scientific_pctx.pth' \
    --run_mode='test'
