CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
    --main_process_port 11223 \
    main.py \
    --model=DuoRec \
    --dataset=AmazonReviews2023 \
    --category=Industrial_and_Scientific \
