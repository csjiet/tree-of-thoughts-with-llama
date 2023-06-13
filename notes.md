-----------
prompts directory:
- crossword.py: 
    - variables: stores few-shot examples, and leaves a placeholder for input and output following the few shot examples.
        - "standard_prompts" (prompts (Input, Output) without "thoughts"), 
        - "cot_prompts" (prompts (Input, Thoughts, Output) with "thoughts")
        - "propose_prompt" (
                "
                Let's play a 5 x 5 mini crossword, where each word should have exactly 5 letters.
                {input}
                Given the current status, list all possible answers for unfilled or changed words, and your confidence levels (certain/high/medium/low), using the format "h1. apple (medium)". Use "certain" cautiously and only when you are 100% sure this is the correct word. You can list more then one possible answer for each word."
        )
        - "value_prompt" (a prompt that follows: 
            Evaluate if there exists a five letter word of some meaning that fit some letter constraints (sure/maybe/impossible).
            for each word word 
            - Definition: crossword box
            - Describes positional letter constraints
            - List possible words that match definition: reason conflict/ matches
            - Reasonining
            - Verdict
        )

- game24.py:
    - variables: stores few-shot examples, and leaves a placeholder for input and output following the few shot examples.
        - "standard_prompts" (prompts with just - "Input", "Output")
        - "cot_prompts" (prompts with Input, Steps, Output - with "thoughts")
        - "propose_prompt" (
            Input (list of numbers)
            possible next steps (which lists 2 operands, 1 operator, and its result, and appending the result to a new list which the remaining operands which are not used)
        )
        - "evaluate_prompt" (
            Evaluate if given numbers can reach 24 (sure/likely/impossible)
            - list of number
            - numbers, operations, results
            - reasoning 
            - Verdict 
        )
        - "value_last_step_prompt" (
            Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
            - Input (Four numbers) 
            - Output/ Answer
            - Judge
            - Verdict

        )
- text.py:
    - standard_prompt = 'Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}'
    - cot_prompt = ' Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}
        Make a plan then write. Your output should be of the following format:
        Plan:
        Your plan here.
        Passage:
        Your passage here.
        '
    - vote_prompt = 'Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.'
    - compare_prompt = 'Briefly analyze the coherency of the following two passages. Conclude in the last line "The more coherent passage is 1", "The more coherent passage is 2", or "The two passages are similarly coherent".'
    - score_prompt = 'Analyze the following passage, then at the last line conclude "Thus the coherency score is {s}", where s is an integer from 1 to 10.'

-----------
data directory
- 24
Rank,Puzzles,AMT (s),Solved rate,1-sigma Mean (s),1-sigma STD (s)

- crosswords 
[ [[ list_of_definitions ], [letters_listed_vertically]], ...]

- text
100 list of sentences

-----------
tasks directory
- base.py
- 24.py
- crossword.py
- text.py






