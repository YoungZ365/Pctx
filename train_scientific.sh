CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
    --main_process_port 11001 \
    main.py \
    --model=Pctx \
    --dataset=AmazonReviews2023 \
    --category=Industrial_and_Scientific \
    --run_GR_or_not=True \
    --rq_faiss=True \
    --frequency_threshold=0.2 \
    --augmentation_probability=0.9
