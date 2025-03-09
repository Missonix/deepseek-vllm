from vllm_wrapper import vLLMWrapper

model = "/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # 使用本地模型路径

vllm_model = vLLMWrapper(model,
                         dtype = 'float16',
                         tensor_parallel_size=2,
                         gpu_memory_utilization=0.70)

history = None

while True:
    Q=input("请输入问题：")
    response, history = vllm_model.chat(query=Q, history=history)

    print(response)
    history = history[:20]