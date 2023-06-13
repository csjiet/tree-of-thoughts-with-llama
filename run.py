import os
import json
import itertools
import argparse
import numpy as np
from functools import partial
from models import gpt, gpt_usage
from model_llama import llama, llama_usage
from tasks import get_task # get_task is a function defined in tasks/__init__.py, where it imports a task class from e.g.: tasks/text.py and calls a constructor for that class to create an object, and returns it. 

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    # value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)

    print('++++++++++++++++++++++++++++++++++++++++++++')
    print('get_value() function')
    print('value_prompt: \n', value_prompt)
    # Replaced with llama
    value_outputs = llama(value_prompt, model = 'llama-7B', max_tokens = 60, do_sample = True, beams =1, n= n_evaluate_sample)

    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value

    print('\noutput: \n', value)
    print('--------------------------------------------')
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    # vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)

    print('++++++++++++++++++++++++++++++++++++++++++++')
    print('get_votes() function')
    print('vote_prompt: \n', vote_prompt)
    # Replaced with llama
    vote_outputs = llama(vote_prompt, model = 'llama-7B', max_tokens = 60, do_sample = True, beams =1, n = n_evaluate_sample)


    values = task.vote_outputs_unwrap(vote_outputs, len(ys))

    print('\noutput: \n', values)
    print('--------------------------------------------')
    return values

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    # proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')

    print('++++++++++++++++++++++++++++++++++++++++++++')
    print('get_proposals() function')
    print('propose_prompt: \n', propose_prompt)
    # Replaced with llama
    proposals = llama(propose_prompt, model = 'llama-7B', max_tokens = 60, do_sample = False, beams = 1, n=1)[0].split('\n')

    print('\noutput: \n',[y + _ + '\n' for _ in proposals])
    print('--------------------------------------------')

    return [y + _ + '\n' for _ in proposals]

# Use wrapped prompts to generate new samples from LLM
# TODO: add support for other sampling methods
def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    # samples = gpt(prompt, n=n_generate_sample, stop=stop)

    print('++++++++++++++++++++++++++++++++++++++++++++')
    print('get_samples() function')
    print('prompt: \n', prompt)
    # Replaced with llama
    samples = llama(prompt, model = 'llama-7B', max_tokens = 60, do_sample = True, beams =1, n = n_generate_sample)

    print('\noutput: \n', [y + _ for _ in samples])
    print('--------------------------------------------')

    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True):
    # print(gpt)
    print('++++++++++++++++++++++++++++++++++++++++++++')
    print('solve() function')
    # print(llama)

    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    
    print('--------------------------------------------')

    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None) # Get generated output from LLM 
    return ys, {}

def run(args):
    task = get_task(args.task, args.task_file_path) # function: parse_args() parses cli by storing each value associated with a flag e.g., --task, --task_file_path, and returns an object which stores value of parsed arguments as attributes, which can be accessed like a field.
    logs, cnt_avg, cnt_any = [], 0, 0
    # global gpt
    # gpt = partial(gpt, model=args.backend, temperature=args.temperature) # partial function creates a new function which takes in a prompt - gpt - with the model and temperature fixed.

    # Replaced with llama
    global llama
    llama = llama 

    if args.naive_run: # create new directory and file name to store generated data
        file = f'logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    else:
        file = f'logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):

        # Run standard prompting, cot, tot
        # solve
        if args.naive_run: # naive run happens to all standard.* and cot prompts,
            ys, info = naive_solve(args, task, i) 
        else:
            ys, info = solve(args, task, i) # bfs

        # Appends a dictionary to logs 
        # log 
        infos = [task.test_output(i, y) for y in ys] # test_output() for each task are defined in ./task/* which returns a dictionary with possible keys 'r', 's' - reward and the solution respectively, etc, that describes the prompt.
        
        # info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': gpt_usage(args.backend)}) # update info dictionary with idx, ys, infos, and usage_so_far

        # Replaced with llama_usage
        info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': llama_usage(args.backend)})

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
    print('usage_so_far', llama_usage(args.backend))


def parse_args():
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
    return args



if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)