import os
import re
from tasks.base import Task, DATA_PATH
from prompts.text import *
from models import gpt
import globals


class TextTask(Task):
    """
    Input (x)   : a text instruction
    Output (y)  : a text generation
    Reward (r)  : # TODO
    Input Example: 
    Output Example: 
    """
    def __init__(self, file='data_100_random_text.txt'):
        """
        file: a text file, each line is some sentences
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'text', file)
        self.data = open(path).readlines() # reads all lines from a file into a list
        self.steps = 2
        self.stops = ['\nPassage:\n', None]

    # Determines the number of lines in the file
    def __len__(self) -> int: 
        return len(self.data)
    
    # Returns the line at the given index
    def get_input(self, idx: int) -> str: 
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        # breakpoint()
        output = output.split('Passage:\n')[-1] # split string in output variable by delimeter 'Passage:\n' and return the last element. Where the last element is the generated text.
        prompt = score_prompt + output # concatenate score_prompt and generated text. score_prompt is defined in prompts/text.py
        # score_outputs = gpt(prompt, n=5, model='gpt-4') # gpt takes a prompt and returns (n=5) - 5 text, stored in score_outputs variable. gpt is defined in models/gpt.py 
        
        # Replaced with llama
        score_outputs = []
        # for _ in range(5): # FIX THIS !
        #     score_outputs.append(llama(prompt, model = 'llama-7B', max_tokens = 10)[0])
        score_outputs = globals.LLM.llama(prompt, max_tokens = 100, do_sample = False, beams = 5, n= 5)

        print(score_outputs)

        scores = []
        for score_output in score_outputs:
            # print(score_output)
            pattern = r".*coherency score is (\d+).*" 
            match = re.match(pattern, score_output, re.DOTALL) # re.match(pattern, string, flags=0) returns a match object on success, None on failure.
            # If match, then match.groups() returns a tuple containing all the subgroups of the match, from 1 up to however many groups are in the pattern. match.groups()[0] returns the first element of the tuple.
            if match:
                score = int(match.groups()[0]) 
                scores.append(score)
            else:
                print(f'------------------score no match: {[score_output]}')
        print(scores)
        # print('------------')
        info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0} # info is a dictionary with keys 'rs' and 'r'. 'rs' is a list of scores, 'r' is the average of the scores.
        return info

    # these static methods are called in run.py
    # .*_wrap functions take a string as input to fit into the prompt placeholder, and generates the final output string
    # .*_unwrap functions take a string or list as input to fit into the prompt placeholder, and generates the final output string 

    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        # breakpoint()
        return standard_prompt.format(input=x) + y # standard_prompt is defined in prompts/text.py and is a string. standard_prompt.format(input=x) returns a string with the input x that is placed in a placeholder in standard_prompt. y is an empty string, allowing room for customization.

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        # breakpoint()
        return cot_prompt.format(input=x) + y # cot_prompt is defined in prompts/text.py and is a string. cot_prompt.format(input=x) returns a string with the input x that is placed in a placeholder in cot_prompt. y is an empty string, allowing room for customization.

    ##############################
    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        # breakpoint()
        prompt = vote_prompt
        for i, y in enumerate(ys, 1): # enumerate(ys, 1) returns a list of tuples. Each tuple contains an index and an element from ys. The index starts at 1.
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f'Choice {i}:\n{y}\n' # concatenate prompt and the choice number and the choice text

        return prompt

    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        # breakpoint()
        vote_results = [0] * n_candidates # vote_results is a list of n_candidates number of 0s
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL) # re.match(pattern, string, flags=0) returns a match object on success, None on failure.
            if match:
                vote = int(match.groups()[0]) - 1 # vote is the first element of the tuple returned by match.groups(). vote is the index of the best choice.
                if vote in range(n_candidates): # NOT a loop: if vote is in the range of 0 to n_candidates, index into vote_results and increment the vote count for the best choice
                    vote_results[vote] += 1 
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results
    ##############################

    # Prepare compare prompt (wrap)
    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        # breakpoint()
        assert len(ys) == 2, 'compare prompt only supports 2 candidates' # ensures that there are only 2 candidates 
        ys = [y.split('Passage:\n')[-1] for y in ys] # For both candidates, extract the generated text after the delimeter 'Passage:\n' (in the cot prompt). 
        prompt = compare_prompt + f'Passage 1:\n{ys[0]}\n\nPassage 2:\n{ys[1]}\n' # concatenate the compare prompt with he two extracted passages from the two candidates, to prepare compare prompt
        return prompt

    # Extract result after compare prompt (unwrap): Assign score for each expected prompt output enforced by compare prompt. 
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        # breakpoint()
        if 'more coherent passage is 1' in compare_output:
            return 0
        elif 'more coherent passage is 2' in compare_output:
            return 1
        elif 'two passages are similarly coherent' in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1
