import sys
import time
import torch
import re
sys.path.insert(0, '/hdd4/zoo/llama/7B')
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

    def llama(self, text_prompt, max_tokens = 1000, do_sample = False, beams = 1, n = 1, top_k = 50, top_p = 1.0, temperature = 1.0): # dataset_name is not used

        output = None

        with torch.no_grad():

            # encode input text prompt as pytorch tensor
            inputs = self.tokenizer(text_prompt, return_tensors='pt') 

            inputs = inputs.to('cuda:0') 
            self.model.to('cuda:0')

            # Decoding: generate output tensor 
            output = self.model.generate(inputs["input_ids"], max_new_tokens= max_tokens, do_sample = do_sample, num_beams = beams, num_return_sequences = n, temperature = temperature, top_k = top_k, top_p = top_p) # https://huggingface.co/docs/transformers/v4.30.0/en/generation_strategies  
            # output = model.module.generate(inputs["input_ids"], max_new_tokens= max_tokens, do_sample = do_sample, num_beams = beams, num_return_sequences = n) # https://huggingface.co/docs/transformers/v4.30.0/en/generation_strategies  
            # output = model.generate(**inputs, max_new_tokens= max_tokens)

        # Decode output tensor into text
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens= True)

        # Free input tensors from GPU memory
        del inputs
        torch.cuda.empty_cache()

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
    llm = LLM(model_name='llama-13B')

    # # Prompt example 1
    # start_time = time.time()
    # prompt = '''
    # Given 4 input numbers labeled A, B, C, D: labeled as \"Input: A B C D\".
    # Select two numbers and an arithmetic operator from the list of operators (+, -, *, /) to form a valid expression. The expression should evaluate to the third number R.
    # For example if A and B are chosen as the operand and * is the operator, it will generate a line: \"A * B = R (left: R C D)\", where C and D are the remaining unused/ left out numbers. 
    # Repeat selection and evaluation of expression and output format, and list it as output 
    # Here is an example in double quotes:
    # \"Input: 2 8 8 14
    # Possible next steps:
    # 2 + 8 = 10 (left: 10 8 14)
    # 8 / 2 = 4 (left: 4 8 14)
    # 14 + 2 = 16 (left: 16 8 8)
    # 2 * 8 = 16 (left: 16 8 14)
    # 8 - 2 = 6 (left: 6 8 14)
    # 14 - 8 = 6 (left: 6 2 8)
    # 14 /  2 = 7 (left: 7 8 8)
    # 14 - 2 = 12 (left: 12 8 8)\"
    # Given the example above, continue the output after the double quotes. 
    # \"Input: 4 5 6 10
    # \"Possible next steps\":\"
    # '''
    # print(f'Prompt: {prompt}')
    # output = llm.llama(prompt, max_tokens = 200)

    # print('-------------------------Output starts: -------------------------------------------')
    # for op in output:
    #     print(op, '\n')
    # print('-------------------------Output ends: -------------------------------------------')

    # end_time = time.time()
    # print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')

    # Prompt example 2
    start_time = time.time()
    prompt = '''\nWrite a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: It isn't difficult to do a handstand if you just stand on your hands. It caught him off guard that space smelled of seared steak. When she didn’t like a guy who was trying to pick her up, she started using sign language. Each person who knows you has a different perception of who you are.\n\n\nMake a plan then write. Your output should be of the following format:\n\nPlan:\nYour plan here.\n\nPassage:\nYour passage here.\n
    '''
    print(f'Prompt: {prompt}') 
    output = llm.llama(prompt, max_tokens = 200)

    print('-------------------------Output starts: -------------------------------------------')
    for op in output:
        print(op, '\n')
    print('-------------------------Output ends: -------------------------------------------')

    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')

    # Prompt example 3
    start_time = time.time()
    prompt = '''
    Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.
    \nChoice 1:\n*/\nChoice 2:\nIt isn\'t difficult to do a handstand if you just stand on your hands.\n\nIt caught him off guard that space smelled of seared steak.\n\nWhen she didn’t like a guy who 
    was trying to pick her up, she started using sign language.\n\nEach person who knows you has a different perception of who you are.\nChoice 3:\nIt isn\'t difficult to do a handstand if you just stand on
    your hands.\n\nIt caught him off guard that space smelled of seared steak.\n\nWhen she didn’t like a guy who was trying to pick her up, she started using sign language.\n\nEach person who knows you has
    a different perception of who you are.\nChoice 4:\nIt isn\'t difficult to do a handstand if you just stand on your hands.\n\nIt caught him off guard that space smelled of seared steak.\n\nWhen she 
    didn’t like a guy who was trying to pick her up, she started using sign language.\n\nEach person who knows you has a different perception of who you are.\n\n*/\nChoice 5:\n\n
    '''
    print(f'Prompt: {prompt}') 
    output = llm.llama(prompt, max_tokens = 200)

    print('-------------------------Output starts: -------------------------------------------')
    for op in output:
        print(op, '\n')
    print('-------------------------Output ends: -------------------------------------------')

    end_time = time.time()
    print(f'\n- Output: {output}\n- Total time (s): {end_time - start_time} \n---------------------------')
