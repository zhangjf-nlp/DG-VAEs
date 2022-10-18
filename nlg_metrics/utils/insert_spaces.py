def insert_spaces(input, tokenize=None):
    """ tokenize the input string and insert spaces between tokens
    Args:
        input (str or list[str] or dict[obj:list[str]]): the input string or a list of string or a dict of string list
        tokenize (callable, optional) : the tokenize api to convert a string to a list of tokens, (default jieba.Tokenizer().tokenize)
    """
    if tokenize is None:
        tokenize = str.split
    
    if type(input) is list:
        return [insert_spaces(_, tokenize) for _ in input]
    elif type(input) is dict:
        return {key:[insert_spaces(_, tokenize) for _ in input[key]] for key in input}
    elif type(input) is str:
        return " ".join([word for word in tokenize(input)])
    else:
        assert False, f"unsupported input type: {type(input)}"