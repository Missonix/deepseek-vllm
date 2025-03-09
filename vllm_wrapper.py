import copy
import os
from vllm import LLM
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig, snapshot_download
from prompt_utils import _build_prompt,remove_stop_words

# 特殊token
IMSTART='<|im_start|>'  
IMEND='<|im_end|>'
ENDOFTEXT='<|endoftext|>'     # EOS以及PAD都是它

class vLLMWrapper:
    def __init__(self, 
                 model_dir,
                 tensor_parallel_size=2,
                 gpu_memory_utilization=0.90,
                 dtype='float16',
                 quantization=None):
        # 模型目录下的generation_config.json文件，是推理的关键参数
        '''
        {
            "do_sample": true,
            "max_length": 32768,
            "temperature": 0.9,
            "top_k": 0,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "transformers_version": "4.31.0"
            }
        '''
        
        # 检查是否为本地路径
        is_local_path = os.path.exists(model_dir)
        
        # 如果不是本地路径，尝试从ModelScope下载
        if not is_local_path:
            # 模型下载
            snapshot_download(model_dir)
        
        self.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
        
        # 加载分词器
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer.eos_token_id=self.generation_config.eos_token_id
        
        # 推理终止词，遇到这些词停止继续推理
        # 检查分词器类型，适配不同的分词器
        self.stop_words_ids = []
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            self.stop_words_ids.append(self.tokenizer.eos_token_id)
        
        # 如果是Qwen系列分词器，添加特殊token
        if hasattr(self.tokenizer, 'im_start_id') and hasattr(self.tokenizer, 'im_end_id'):
            self.stop_words_ids.extend([self.tokenizer.im_start_id, self.tokenizer.im_end_id])
            
        # vLLM加载模型
        os.environ['VLLM_USE_MODELSCOPE'] = 'True' if not is_local_path else 'False'
        self.model=LLM(model=model_dir,
                       tokenizer=model_dir,
                       tensor_parallel_size=tensor_parallel_size,
                       trust_remote_code=True,
                       quantization=quantization,
                       gpu_memory_utilization=gpu_memory_utilization, # 0.6
                       dtype=dtype)

    def chat(self,query,history=None,system="You are a helpful assistant.",extra_stop_words_ids=[]):
        # 历史聊天
        if history is None:
            history = []
        else:
            history = copy.deepcopy(history)

        # 额外指定推理停止词
        stop_words_ids=self.stop_words_ids+extra_stop_words_ids

        # 构造prompt
        prompt_text,prompt_tokens=_build_prompt(self.generation_config,self.tokenizer,query,history=history,system=system)
        
        # 打开注释，观测底层Prompt构造
        # print(prompt_text)

        # VLLM请求配置
        sampling_params=SamplingParams(stop_token_ids=stop_words_ids, 
                                         # early_stopping=False,  # 该参数在当前vLLM版本中不支持
                                         top_p=self.generation_config.top_p,
                                         top_k=-1 if self.generation_config.top_k == 0 else self.generation_config.top_k,
                                         temperature=self.generation_config.temperature,
                                         repetition_penalty=self.generation_config.repetition_penalty,
                                         max_tokens=self.generation_config.max_new_tokens)
        
        # 调用VLLM执行推理（批次大小1）
        # vLLM 0.7.3 支持直接使用字符串作为输入
        req_outputs = self.model.generate(prompts=[prompt_text], sampling_params=sampling_params, use_tqdm=False)
        req_output = req_outputs[0]        
        
        # 移除停用词        
        response_token_ids=remove_stop_words(req_output.outputs[0].token_ids,stop_words_ids)
        response=self.tokenizer.decode(response_token_ids)

        # 整理历史对话
        history.append((query,response))
        return response,history