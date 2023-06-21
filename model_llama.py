import sys
import time
import torch
import re
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
        self.model = LlamaForCausalLM.from_pretrained(f'/hdd4/zoo/llama/{model_dir_name}', torch_dtype = torch.float16) # https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map

    def llama(self, text_prompt, max_tokens = 1000, do_sample = False, beams = 1, n = 1, top_k = 50, top_p = 0.95, temperature = 1.0): # dataset_name is not used

        output = None

        with torch.no_grad():

            # encode input text prompt as pytorch tensor
            inputs = self.tokenizer(text_prompt, return_tensors='pt') 

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
            output = self.model.generate(inputs["input_ids"], max_new_tokens= max_tokens, do_sample = do_sample, num_beams = beams, num_return_sequences = n, temperature = temperature) # https://huggingface.co/docs/transformers/v4.30.0/en/generation_strategies  
            # output = model.module.generate(inputs["input_ids"], max_new_tokens= max_tokens, do_sample = do_sample, num_beams = beams, num_return_sequences = n) # https://huggingface.co/docs/transformers/v4.30.0/en/generation_strategies  
            # output = model.generate(**inputs, max_new_tokens= max_tokens)

        # Decode output tensor into text
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens= True)

        # Free input tensors from GPU memory
        del inputs

        torch.cuda.empty_cache()

        # print(decoded_output)
        # String processing: remove text prompt and extract the response from decoded output
        op = []
        for resp in decoded_output:
            cleaned_op = resp.replace(text_prompt, '')
            op.append(cleaned_op.strip())

        return op

    def llama_usage(self, backend='7B'):
        return {'completion_tokens': 0, 'prompt_tokens': 0, 'cost': 0}


if __name__ == '__main__':

    # Load the pretrained model once
    llm = LLM(model_name='llama-7B')


    # Prompt example 1
    start_time = time.time()
    prompt = '''
    Given 4 input numbers labeled A, B, C, D: labeled as \"Input: A B C D\".
    Select two numbers and an arithmetic operator from the list of operators (+, -, *, /) to form a valid expression. The expression should evaluate to the third number R.
    For example if A and B are chosen as the operand and * is the operator, it will generate a line: \"A * B = R (left: R C D)\", where C and D are the remaining unused/ left out numbers. 
    Repeat selection and evaluation of expression and output format, and list it as output 
    Here is an example in double quotes:
    \"Input: 2 8 8 14
    Possible next steps:
    2 + 8 = 10 (left: 10 8 14)
    8 / 2 = 4 (left: 4 8 14)
    14 + 2 = 16 (left: 16 8 8)
    2 * 8 = 16 (left: 16 8 14)
    8 - 2 = 6 (left: 6 8 14)
    14 - 8 = 6 (left: 6 2 8)
    14 /  2 = 7 (left: 7 8 8)
    14 - 2 = 12 (left: 12 8 8)\"
    Given the example above, continue the output after the double quotes. 
    \"Input: 4 5 6 10
    \"Possible next steps\":\"
    '''
    print(f'Prompt: {prompt}')
    output = llm.llama(prompt, max_tokens = 200)
    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')

    # Prompt example 2
    start_time = time.time()
    prompt = 'If there is a robbery in the park, and Bob is one of two men in the park? What is the probability that Bob committed the robbery?'
    print(f'Prompt: {prompt}')
    output = llm.llama(prompt, max_tokens = 200, do_sample = False, beams= 1, n = 1)
    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')

    # Prompt example 3
    start_time = time.time()
    prompt = 'If there is a robbery in the park, and Bob is one of two men in the park? What is the probability that Bob committed the robbery?'
    print(f'Prompt: {prompt}')
    output = llm.llama(prompt, max_tokens = 200, do_sample = False, beams= 2, n = 2)
    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')


    # Prompt example 4
    start_time = time.time()
    prompt = '''Evaluate if given numbers can reach 24 (sure/likely/impossible)
    10 14
    10 + 14 = 24
    sure
    11 12
    11 + 12 = 23
    12 - 11 = 1
    11 * 12 = 132
    11 / 12 = 0.91
    impossible
    4 4 10
    4 + 4 + 10 = 8 + 10 = 18
    4 * 10 - 4 = 40 - 4 = 36
    (10 - 4) * 4 = 6 * 4 = 24
    sure
    4 9 11
    9 + 11 + 4 = 20 + 4 = 24
    sure
    5 7 8
    5 + 7 + 8 = 12 + 8 = 20
    (8 - 5) * 7 = 3 * 7 = 21
    I cannot obtain 24 now, but numbers are within a reasonable range
    likely
    5 6 6
    5 + 6 + 6 = 17
    (6 - 5) * 6 = 1 * 6 = 6
    I cannot obtain 24 now, but numbers are within a reasonable range
    likely
    10 10 11
    10 + 10 + 11 = 31
    (11 - 10) * 10 = 10
    10 10 10 are all too big
    impossible
    1 3 3
    1 * 3 * 3 = 9
    (1 + 3) * 3 = 12
    1 3 3 are all too small
    impossible
    9 10 10 
    '''
    print(f'Prompt: {prompt}')
    output = llm.llama(prompt, max_tokens = 200, do_sample = False, beams= 1, n = 1)
    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')
