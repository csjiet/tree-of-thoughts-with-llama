def get_task(name, file=None):
    if name == 'game24':
        from .game24 import Game24Task
        return Game24Task(file) # calling the constructor defined in game24.py as __init__()
    elif name == 'text':
        from .text import TextTask
        return TextTask(file) # calling the constructor defined in text.py as __init__()
    elif name == 'crosswords':
        from .crosswords import MiniCrosswordsTask
        return MiniCrosswordsTask(file)
    else:
        raise NotImplementedError
