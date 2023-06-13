import sys
import time
import torch
sys.path.insert(0, '/hdd4/zoo/llama/7B')

from prompts.text import *
from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import load_checkpoint_and_dispatch
from optimum.bettertransformer import BetterTransformer

def llama(text_prompt, model = 'llama-7B', max_tokens = 1000, do_sample = False, beams = 1, n = 1, top_k = 50, top_p = 0.95): # dataset_name is not used

    start_time = time.time()
    model_dir_name = model.split('-')[-1] # e.g.,: 'llama-7B' -> '7B'

    # load tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(f'/hdd4/zoo/llama/{model_dir_name}') # converts text: tokenization, numerical encoding, attention masking - into suitable input for the LLM
    # model = LlamaForCausalLM.from_pretrained(f'/hdd4/zoo/llama/{model_dir_name}') # https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map
    model = LlamaForCausalLM.from_pretrained(f'/hdd4/zoo/llama/{model_dir_name}', torch_dtype = torch.float16) # https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map

    # model.tie_weights()
    # model = load_checkpoint_and_dispatch(model,f'/hdd4/zoo/llama/{model_dir_name}')

    output = None

    # encode input text prompt as pytorch tensor
    inputs = tokenizer(text_prompt, return_tensors='pt') 
    print(inputs)

    with torch.no_grad():

        # TODO: Might need to break text prompt into smaller batches before feeding into gpu 
        # to prevent CUDA out of memory error


        # if torch.cuda.device_count() > 1:
            # model = torch.nn.DataParallel(model) # enable data parallelism # https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/data_parallel_tutorial.py


        # TODO: How to optimize a deep learning model for faster inference? https://www.thinkautonomous.ai/blog/deep-learning-optimization/

        # TODO: Reduce precision of model weights and inputs. Set device_map = 'auto'

        inputs = inputs.to('cuda:0') # https://huggingface.co/docs/accelerate/usage_guides/big_modeling
        model.to('cuda:0')

        # output = model.generate(inputs.input_ids, max_new_tokens= max_tokens)

        # Decoding: generate output tensor 
        output = model.generate(inputs["input_ids"], max_new_tokens= max_tokens, do_sample = do_sample, num_beams = beams, num_return_sequences = n) # https://huggingface.co/docs/transformers/v4.30.0/en/generation_strategies  
        # output = model.module.generate(inputs["input_ids"], max_new_tokens= max_tokens, do_sample = do_sample, num_beams = beams, num_return_sequences = n) # https://huggingface.co/docs/transformers/v4.30.0/en/generation_strategies  
        # output = model.generate(**inputs, max_new_tokens= max_tokens)

        # print(output) # print output tensor

        # Decode output tensor into text
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
    end_time = time.time()# May be included into cost for llama_usage()

    # Free input tensors from GPU memory
    del inputs

    # Free model from GPU memory
    del model

    torch.cuda.empty_cache()

    # String processing: extract response from decoded output
    op = []
    for resp in decoded_output:
        # print('RESP')
        # print(resp.strip().split('\n', 1))
        
        uncleaned_resp = resp.strip().split('\n', 1)
        if len(uncleaned_resp) > 1:
            op.append((uncleaned_resp[-1]).strip())

    return op

def llama_usage(backend='7B'):
    return {'completion_tokens': 0, 'prompt_tokens': 0, 'cost': 0}


if __name__ == '__main__':
    start_time = time.time()
    prompt = 'If there is a robbery in the park, and Bob is one of two men in the park? What is the probability that Bob committed the robbery?'
    print(f'Prompt: {prompt}')
    output = llama(prompt, model = 'llama-7B', max_tokens = 40, do_sample = False, beams= 1, n = 1)
    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')

    start_time = time.time()
    prompt = 'If there is a robbery in the park, and Bob is one of two men in the park? What is the probability that Bob committed the robbery?'
    print(f'Prompt: {prompt}')
    output = llama(prompt, model = 'llama-7B', max_tokens = 40, do_sample = False, beams= 1, n = 1)
    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')
    #Example raw decoded output: ['If there is a robbery in the park, and Bob is one of two men in the park? What is the probability that Bob committed the robbery?\nasked by Tiffany on September 21, 2010\nasked by Tiffany on September 9,']

    start_time = time.time()
    prompt = 'If there is a robbery in the park, and Bob is one of two men in the park? What is the probability that Bob committed the robbery?'
    print(f'Prompt: {prompt}')
    output = llama(prompt, model = 'llama-7B', max_tokens = 40, do_sample = True, beams= 2, n = 1)
    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')

    start_time = time.time()
    prompt = 'If there is a robbery in the park, and Bob is one of two men in the park? What is the probability that Bob committed the robbery?'
    print(f'Prompt: {prompt}')
    output = llama(prompt, model = 'llama-7B', max_tokens = 40, do_sample = True, beams= 2, n = 2)
    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')

    start_time = time.time()
    prompt = 'If there is a robbery in the park, and Bob is one of two men in the park? What is the probability that Bob committed the robbery?'
    print(f'Prompt: {prompt}')
    output = llama(prompt, model = 'llama-7B', max_tokens = 40, do_sample = True, beams= 2, n = 2)
    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')

