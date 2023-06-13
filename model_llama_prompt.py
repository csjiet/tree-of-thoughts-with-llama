text_prompts = {
    "data1": {
        'question_llama': ['List the characteristics of waterbirds: ', 'List the characteristics of landbirds: '],
        'forbidden_words': None
    },
    "data2": {
        'question_llama': ['List visual characteristics of a blonde haired person: ', 'List visual characteristics of dark haired person: '],
        'forbidden_words': ['blonde hair', 'dark hair']
    },
    "data3": {
        'question_llama': [['List visual characteristics of a dog: ', 
                           'List visual characteristics of an elephant: ',
                           'List visual characteristics of a giraffe: ', 
                           'List visual characteristics of a guitar: ',
                           'List visual characteristics of a horse: ',
                           'List visual characteristics of a house: ',
                           'List visual characteristics of a person: ',],
                           ['List spurious/biased characteristics of a dog: ', 
                           'List spurious/biased characteristics of an elephant: ',
                           'List spurious/biased characteristics of a giraffe: ', 
                           'List spurious/biased characteristics of a guitar: ',
                           'List spurious/biased characteristics of a horse: ',
                           'List spurious/biased characteristics of a house: ',
                           'List spurious/biased characteristics of a person: ',],
                           ],
        'forbidden_words': None,
        # ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'],
    }
}