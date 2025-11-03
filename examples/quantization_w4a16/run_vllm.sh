model_path="/data5/yliu7/tmp/Qwen2.5-0.5B-W4A16-G128"
model_path="/data5/yliu7/tmp/Meta-Llama-3.1-8B-Instruct-W4A16-G128"
model_path="/data5/yliu7/tmp/Meta-Llama-3.1-8B-Instruct-W4A16-G128-with-shuffule"
model_path="/data5/yliu7/tmp/Meta-Llama-3.1-8B-Instruct-W4A16-G128-disbale-shuffule/"
# model_path="/data5/yliu7/meta-llama/meta-llama/Meta-Llama-3.1-8B-Instruct-AR-W4G128"
model_path="/storage/yiliu7/Meta-Llama-3.1-8B-Instruct-W4A16-G128-disbale-shuffule"
tp_size=1
VLLM_USE_DEEP_GEMM=0 \
VLLM_LOGGING_LEVEL=DEBUG  \
vllm serve $model_path \
    --max-model-len 8192 \
    --max-num-batched-tokens 32768 \
    --tensor-parallel-size $tp_size \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.8 \
    --dtype bfloat16 \
    --port 8099 \
    --no-enable-prefix-caching \
    --trust-remote-code  2>&1 | tee $log_file




# curl -X POST http://127.0.0.1:8099/v1/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#            "model": "/data5/yliu7/tmp/Qwen2.5-0.5B-W4A16-G128",
#            "prompt": "Solve the following math problem step by step: What is 25 + 37?",
#            "max_tokens": 100,
#            "temperature": 0.7,
#            "top_p": 1.0
#          }'


# curl -X POST http://127.0.0.1:8000/v1/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#            "model": "/data5/yliu7/tmp/Qwen2.5-0.5B-W4A16-G128",
#            "prompt": "Solve the following math problem step by step: What is 25 + 37?",
#            "max_tokens": 100,
#            "temperature": 0.7,
#            "top_p": 1.0
#          }'



# batch_size=4
# lm_eval --model local-completions \
#     --tasks $task_name \
#     --model_args model=${model_path},base_url=http://127.0.0.1:8687/v1/completions,max_concurrent=1,max_length=${max_length},max_gen_toks=${max_gen_toks} \
#     --batch_size ${batch_size}  \
#     --gen_kwargs="max_length=${max_length},max_gen_toks=${max_gen_toks}" \
#     --confirm_run_unsafe_code \
#     --log_samples \
#     --limit 16 \
#     --output_path "benchmark_logs/$EVAL_LOG_NAME" \
#     2>&1 | tee "benchmark_logs/${EVAL_LOG_NAME}.log"