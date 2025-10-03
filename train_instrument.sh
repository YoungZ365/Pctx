CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
    --main_process_port 11002 \
    main.py \
    --model=Pctx \
    --dataset=AmazonReviews2023 \
    --category=Musical_Instruments \
    --run_GR_or_not=True \
    --rq_faiss=True \
    --n_inference_ensemble=5 \
    --frequency_threshold=0.2 \
    --augmentation_probability=0.6
