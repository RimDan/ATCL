#!/bin/bash
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMzQyNTI4ZjgtMTY5MC00ZjY5LThiYWItN2JhOWVlMWEyMTk2In0="

DATA='../data/wiki-103/'
DATASET='wt103'


python3.8 -tt train_atcl.py \
        --cuda \
        --data=$DATA \
        --dataset=$DATASET \
        --adaptive \
        --n_layer=6 \
        --d_model=256 \
        --n_head=1 \
        --d_head=40 \
        --d_inner=2100 \
        --dropout=0.1 \
        --dropatt=0.0 \
        --optim='adam' \
        --lr=0.00025 \
        --max_step=200000 \
        --tgt_len=80 \
        --mem_len=0 \
        --eval_tgt_len=80 \
        --batch_size=45 \
        --gpu0_bsz=1 \
        --adversarial \
