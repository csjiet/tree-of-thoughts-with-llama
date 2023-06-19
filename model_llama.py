import sys
import time
import torch
sys.path.insert(0, '/hdd4/zoo/llama/7B')
sys.path.insert(0, '/hdd4/zoo/llama/13B')
sys.path.insert(0, '/hdd4/zoo/llama/30B')
sys.path.insert(0, '/hdd4/zoo/llama/65B')

from prompts.text import *
from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import load_checkpoint_and_dispatch
from optimum.bettertransformer import BetterTransformer


class LLM: 
    def __init__(self, model_name = 'llama-7B'):

        model_dir_name = model_name.split('-')[-1] # e.g.,: 'llama-7B' -> '7B'

        # load tokenizer and model
        self.tokenizer = LlamaTokenizer.from_pretrained(f'/hdd4/zoo/llama/{model_dir_name}') # converts text: tokenization, numerical encoding, attention masking - into suitable input for the LLM
        # model = LlamaForCausalLM.from_pretrained(f'/hdd4/zoo/llama/{model_dir_name}') # https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map
        self.model = LlamaForCausalLM.from_pretrained(f'/hdd4/zoo/llama/{model_dir_name}', torch_dtype = torch.float16) # https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map

    def llama(self, text_prompt, max_tokens = 1000, do_sample = False, beams = 1, n = 1, top_k = 50, top_p = 0.95): # dataset_name is not used

        output = None

        with torch.no_grad():

            # model.tie_weights()
            # model = load_checkpoint_and_dispatch(model,f'/hdd4/zoo/llama/{model_dir_name}')


            # encode input text prompt as pytorch tensor
            inputs = self.tokenizer(text_prompt, return_tensors='pt') 
            print(inputs)

            # TODO: Might need to break text prompt into smaller batches before feeding into gpu 
            # to prevent CUDA out of memory error


            # if torch.cuda.device_count() > 1:
                # model = torch.nn.DataParallel(model) # enable data parallelism # https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/data_parallel_tutorial.py


            # TODO: How to optimize a deep learning model for faster inference? https://www.thinkautonomous.ai/blog/deep-learning-optimization/

            # TODO: Reduce precision of model weights and inputs. Set device_map = 'auto'

            inputs = inputs.to('cuda:0') # https://huggingface.co/docs/accelerate/usage_guides/big_modeling
            self.model.to('cuda:0')

            # output = model.generate(inputs.input_ids, max_new_tokens= max_tokens)

            # Decoding: generate output tensor 
            output = self.model.generate(inputs["input_ids"], max_new_tokens= max_tokens, do_sample = do_sample, num_beams = beams, num_return_sequences = n) # https://huggingface.co/docs/transformers/v4.30.0/en/generation_strategies  
            # output = model.module.generate(inputs["input_ids"], max_new_tokens= max_tokens, do_sample = do_sample, num_beams = beams, num_return_sequences = n) # https://huggingface.co/docs/transformers/v4.30.0/en/generation_strategies  
            # output = model.generate(**inputs, max_new_tokens= max_tokens)

        # Decode output tensor into text
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens= True)

        # Free input tensors from GPU memory
        del inputs

        torch.cuda.empty_cache()

        # String processing: extract response from decoded output
        op = []
        for resp in decoded_output:
            uncleaned_resp = resp.strip().split('\n', 1)
            if len(uncleaned_resp) > 1:
                op.append((uncleaned_resp[-1]).strip())

        return op

    def llama_usage(self, backend='7B'):
        return {'completion_tokens': 0, 'prompt_tokens': 0, 'cost': 0}


if __name__ == '__main__':

    llm = LLM(model_name='llama-7B')

    prompt = "If given this example:  \"Input: 2 8 8 14 Possible next steps: 2 + 8 = 10 (left: 8 10 14) 8 / 2 = 4 (left: 4 8 14) 14 + 2 = 16 (left: 8 8 16) 2 * 8 = 16 (left: 8 14 16) 8 - 2 = 6 (left: 6 8 14) 14 - 8 = 6 (left: 2 6 8) 14 /  2 = 7 (left: 7 8 8) 14 - 2 = 12 (left: 8 8 12), What is the remaining output for the \"Possible next steps\", given the input - Input: 4 5 6 10  Possible next steps: "
    start_time = time.time()
    # output = llm.llama(prompt, max_tokens = 100, do_sample = False, beams= 1, n = 1)
    output = llm.llama(prompt)
    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')

    prompt = '''
    Input: 2 8 8 14\n
    Possible next steps:\n
    2 + 8 = 10 (left: 8 10 14)\n
    8 / 2 = 4 (left: 4 8 14)\n
    14 + 2 = 16 (left: 8 8 16)\n
    2 * 8 = 16 (left: 8 14 16)\n
    8 - 2 = 6 (left: 6 8 14)\n
    14 - 8 = 6 (left: 2 6 8)\n
    14 /  2 = 7 (left: 7 8 8)\n
    14 - 2 = 12 (left: 8 8 12)\n
    Input: 4 5 6 10\n
    Possible next steps:\n'?
    '''
    start_time = time.time()
    output = llm.llama(prompt)
    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')
    '''
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

    '''