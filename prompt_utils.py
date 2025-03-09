import copy 

# 通用构造prompt的函数
def _build_prompt(
                generation_config,
                tokenizer,
                query,
                history=None,
                system=""):
    # 检查分词器类型，处理不同的模型格式
    if hasattr(tokenizer, 'im_start_id') and hasattr(tokenizer, 'im_end_id'):
        # Qwen格式的构造方式
        return _build_prompt_qwen(generation_config, tokenizer, query, history, system)
    else:
        # 通用/Llama格式的构造方式
        return _build_prompt_llama(generation_config, tokenizer, query, history, system)

# 按chatml格式构造千问(Qwen)的Prompt
def _build_prompt_qwen(
                generation_config,
                tokenizer,
                query,
                history=None,
                system=""):
    if history is None:
        history=[]

    # 包裹发言内容的token
    im_start,im_start_tokens='<|im_start|>',[tokenizer.im_start_id]
    im_end,im_end_tokens='<|im_end|>',[tokenizer.im_end_id]
    # 换行符token
    nl_tokens=tokenizer.encode("\n")

    # 用于编码system/user/assistant的一段发言, 格式{role}\n{content}
    def _tokenize_str(role,content): # 返回元组，下标0是文本，下标1是token ids
        return f"{role}\n{content}",tokenizer.encode(role)+nl_tokens+tokenizer.encode(content)
    
    # 剩余token数
    max_window_size = getattr(generation_config, 'max_window_size', 8192)
    left_token_space = max_window_size

    # prompt头部: system发言
    system_text_part,system_tokens_part=_tokenize_str("system", system) # system_tokens_part -->    system\nYou are a helpful assistant.
    system_text=f'{im_start}{system_text_part}{im_end}'
    system_tokens=im_start_tokens+system_tokens_part+im_end_tokens # <|im_start|>system\nYou are a helpful assistant.<|im_end|>
    left_token_space-=len(system_tokens)
    
    # prompt尾部: user发言和assistant引导
    query_text_part,query_tokens_part=_tokenize_str('user', query)
    query_tokens_prefix=nl_tokens+ im_start_tokens
    query_tokens_suffix=im_end_tokens+nl_tokens+im_start_tokens+tokenizer.encode('assistant')+nl_tokens
    if len(query_tokens_prefix)+len(query_tokens_part)+len(query_tokens_suffix)>left_token_space: # query太长截断
        query_token_len=left_token_space-len(query_tokens_prefix)-len(query_tokens_suffix)
        query_tokens_part=query_tokens_part[:query_token_len]
        query_text_part=tokenizer.decode(query_tokens_part)
    query_tokens=query_tokens_prefix+query_tokens_part+query_tokens_suffix
    query_text=f"\n{im_start}{query_text_part}{im_end}\n{im_start}assistant\n"
    left_token_space-=len(query_tokens)
    
    # prompt腰部: 历史user+assitant对话
    history_text,history_tokens='',[]
    for hist_query,hist_response in reversed(history):    # 优先采用最近的对话历史
        hist_query_text,hist_query_tokens_part=_tokenize_str("user",hist_query) # user\n历史提问
        hist_response_text,hist_response_tokens_part=_tokenize_str("assistant",hist_response) # assistant\n历史回答
        # 生成本轮对话
        cur_history_tokens=nl_tokens+im_start_tokens+hist_query_tokens_part+im_end_tokens+nl_tokens+im_start_tokens+hist_response_tokens_part+im_end_tokens
        cur_history_text=f"\n{im_start}{hist_query_text}{im_end}\n{im_start}{hist_response_text}{im_end}"
        # 储存多轮对话
        if len(cur_history_tokens)<=left_token_space:
            history_text=cur_history_text+history_text
            history_tokens=cur_history_tokens+history_tokens
            left_token_space-=len(cur_history_tokens)
        else:
            break 
            
    # 生成完整Prompt
    prompt_str=f'{system_text}{history_text}{query_text}'
    prompt_tokens=system_tokens+history_tokens+query_tokens
    return prompt_str,prompt_tokens

# 使用Llama系列模型格式构造prompt
def _build_prompt_llama(
                generation_config,
                tokenizer,
                query,
                history=None,
                system=""):
    if history is None:
        history=[]
    
    # 构建完整的文本
    messages = []
    
    # 添加系统消息
    if system:
        messages.append({"role": "system", "content": system})
    
    # 添加历史对话
    for hist_query, hist_response in history:
        messages.append({"role": "user", "content": hist_query})
        messages.append({"role": "assistant", "content": hist_response})
    
    # 添加当前问题
    messages.append({"role": "user", "content": query})
    
    # 添加助手角色前缀
    messages.append({"role": "assistant", "content": ""})
    
    # 尝试使用模型自带的聊天模板 (如果有)
    prompt_text = ""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except:
            # 如果模型没有聊天模板或应用失败，使用基本格式
            pass
    
    # 如果应用模板失败，使用简单格式
    if not prompt_text:
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt_text += f"System: {content}\n\n"
            elif role == "user":
                prompt_text += f"User: {content}\n\n"
            elif role == "assistant":
                if content:
                    prompt_text += f"Assistant: {content}\n\n"
                else:
                    prompt_text += f"Assistant: "
    
    # 转换为token
    prompt_tokens = tokenizer.encode(prompt_text)
    
    return prompt_text, prompt_tokens

# 停用词清理
def remove_stop_words(token_ids,stop_words_ids):
    # 将tuple转换为list，以支持修改操作
    token_ids=list(token_ids) if isinstance(token_ids, tuple) else copy.deepcopy(token_ids)
    while len(token_ids)>0:
        if token_ids[-1] in stop_words_ids:
            token_ids.pop(-1)
        else:
            break
    return token_ids