#!/bin/bash
cd /mnt/d/distill && source venv/bin/activate
vllm serve checkpoints/dpo-awq --quantization awq_marlin --dtype float16 --max-model-len 768 --gpu-memory-utilization 0.95 --enforce-eager --swap-space 0 --max-num-seqs 4 --port 8001 &
sleep 60
uvicorn extractor.api.main:app --host 0.0.0.0 --port 8080
