# Launching servers from port e.g. 8000, 8001, 8002, ...
launch(){
    NUM_GPUS=$1 # number of servers to setup: one server per GPU
    MODEL=$2 # llama3, llama3.1, ...
    FIRST_PORT=$3 # The first starting port of the servers
    FIRST_GPU=$4 # The first starting GPU idx to launch

    # Available models on the server, default is llama3.1
    if [ $MODEL == "llama3.2-1B-Instruct" ]; then
        MODEL_PATH=meta-llama/Llama-3.2-1B-Instruct
        #MODEL_PATH=/home/huidong/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct
        CHAT_TEMPLATE=src/my_template.jinja
    elif [ $MODEL == "llama3.1" ]; then
        MODEL_PATH="TODO"
    else
        MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
    fi

    for idx in $( seq 0 $((NUM_GPUS - 1)) )
    do
        GPU_ID=$((idx + FIRST_GPU))  # The GPU idx for the current server
        PORT=$((idx + FIRST_PORT)) # The port for the current server
        export CUDA_VISIBLE_DEVICES=$GPU_ID # Set the CUDA environment variable

        echo "Starting server on port $PORT with GPU $GPU_ID"
        # vllm serve $MODEL_PATH \
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --port "$PORT" \
            --gpu_memory_utilization 0.6 \
            --dtype bfloat16 \
            --max_model_len 8192 \
            --disable-log-requests &
    done
}

launch 3 llama3.2-1B-Instruct 8000 0