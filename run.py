import os
import json
import itertools
import argparse
import numpy as np
from functools import partial
from models import gpt, gpt_usage
# from model_llama import llama, llama_usage
import model_llama
from tasks import get_task # get_task is a function defined in tasks/__init__.py, where it imports a task class from e.g.: tasks/text.py and calls a constructor for that class to create an object, and returns it. 

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    # breakpoint()
    value_prompt = task.value_prompt_wrap(x, y)

    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    # value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)

    # Replaced with llama
    value_outputs = LLM.llama(value_prompt, max_tokens = 100, do_sample = False, beams = n_evaluate_sample, n= n_evaluate_sample)

    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value

    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    # breakpoint()
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        # breakpoint()
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    # breakpoint()
    vote_prompt = task.vote_prompt_wrap(x, ys)
    # vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)

    # Replaced with llama
    vote_outputs = LLM.llama(vote_prompt, max_tokens = 100, do_sample = False, beams = n_evaluate_sample, n = n_evaluate_sample)


    values = task.vote_outputs_unwrap(vote_outputs, len(ys))

    return values

def get_proposals(task, x, y): 
    # breakpoint()
    propose_prompt = task.propose_prompt_wrap(x, y) 
    # proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')

    # Replaced with llama
    proposals = LLM.llama(propose_prompt, max_tokens = 100, do_sample = False, beams = 1, n= 1)[0].split('\n')

    return [y + _ + '\n' for _ in proposals]

# Use wrapped prompts to generate new samples from LLM
# TODO: add support for other sampling methods
def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    # breakpoint()
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    # samples = gpt(prompt, n=n_generate_sample, stop=stop)

    # Replaced with llama
    samples = LLM.llama(prompt, max_tokens = 100, do_sample = False, beams = n_generate_sample, n = n_generate_sample)

    return [y + _ for _ in samples]

# Arguments (e.g., from bash script/game24/bfs.sh)
# args: Namespace(backend='gpt-4', temperature=0.7, task='game24', task_file_path='24.csv', task_start_index=900, task_end_index=1000, naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)
# task: <tasks.game24.Game24Task object at 0x7fe16037fe50>
# idx: 900
def solve(args, task, idx, to_print=True):
    # breakpoint()
    # print(gpt)

    x = task.get_input(idx)  # p: '4 5 6 10' - from 24.csv, read as a pandas frame, extracting 'Puzzles' column, and then indexing into the 900th puzzle
    ys = [''] 
    infos = []

    # Breadth of tree in bfs ToT is set using cli: --n_generate_sample
    # Height of tree in ToT is set using task.steps in their respective tasks/{file}.py files
    for step in range(task.steps): # p: (task.steps = 4 for game24.py) - Set manually in task/{files}.py - e.g., task.steps for game24.py is 4 for 4 operations; text.py is 2.; crossword.py is 10 steps.
        # breakpoint()
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys)) # itertools.chain takes iterables and convert to one iterable
        ids = list(range(len(new_ys)))
        # breakpoint()
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # breakpoint()
        # selection - bfs/ dfs are greedy - essentially, based on the values in evaluation, 
        # For greedy, we select the top n_select_sample 
        # For sample, we select n_select_sample based on the probability distribution of the values, 
        # where we fix the size of output: because 'size' argument is the output shape of random samples of numpy array.
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values) # Convert 'values' assigned to each response to probability distribution
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist() # Randomly select n_select_sample for each ys identified by their ids, based on the probability distribution ps (which corresponds to each id/ ys)
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids] # using the filtered identifier ids (select_ids), select the corresponding new_ys, and assign to select_new_ys

        # breakpoint()
        # log
        if to_print: 
            # Sort the values and new_ys based on the values
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

        # Append the information of each step to the json file 
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    # breakpoint()
    if to_print: 
        print(ys)

    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    # breakpoint()
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None) # Get generated output from LLM 
    return ys, {}

def run(args):
    # Ensures functions invoked using 'task' - an object of a Task class - are from their respective class where they are defined.
    task = get_task(args.task, args.task_file_path) # returns a task class object (e.g., Game24Task class in tasks/game24.py) returned by get_task() which is imported as a function from __init__.py, which takes in args.task (identifier of task entered in the cli) and returns an instantiated object of a task class e.g., Game24Task class obj in tasks/game24.py
    logs, cnt_avg, cnt_any = [], 0, 0
    # global gpt
    # gpt = partial(gpt, model=args.backend, temperature=args.temperature) # partial function creates a new function which takes in a prompt - gpt - with the model and temperature fixed.

    # Replaced with llama
    global LLM 
    
    LLM = model_llama.LLM(model_name='llama-13B')

    # breakpoint()
    if args.naive_run: # create new directory and file name to store generated data
        file = f'logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):

        # breakpoint()
        # solve: choosing between standard prompting, CoT, ToT
        if args.naive_run: # naive run happens to all standard.* and cot prompts,
            ys, info = naive_solve(args, task, i) 
        else:
            ys, info = solve(args, task, i) # bfs

        # breakpoint()
        # Appends a dictionary to logs 
        # log 
        infos = [task.test_output(i, y) for y in ys] # test_output() for each task are defined in ./task/* 
        
        # info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': gpt_usage(args.backend)}) # update info dictionary with idx, ys, infos, and usage_so_far

        # Replaced with llama_usage
        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': LLM.llama_usage(args.backend)})

        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)
        
        # log main metric
        accs = [info['r'] for info in infos]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        print(i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
    
    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)

    # print('usage_so_far', gpt_usage(args.backend))

    # Replaced with llama_usage
    print('usage_so_far', LLM.llama_usage(args.backend))


def parse_args():
    # breakpoint()
    args = argparse.ArgumentParser()
    # TODO: choices should change to reflect new LLM - llama
    # args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo'], default='gpt-4') # enforces arguments followed by flag to be 'gpt-4' or 'gpt-3.5-turbo'

    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'llama-7B','llama-13B','llama-30B','llama-65B'], default='gpt-4') 
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords'])
    args.add_argument('--task_file_path', type=str, required=True)
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'])
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)

    args = args.parse_args() 
    # breakpoint()
    return args



if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)