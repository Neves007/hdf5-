import torch
import re

def TP_to_class(TP):
    return TP.max(-1)[1]

def transition_to_latex(origion_state:str, target_state:str):
    '''
    将迁移转为latex
    :param origion_state:
    :param target_state:
    :return:
    '''

    def replace_numbers_with_underscore(text):
        return re.sub(r'\d+', lambda match: f"_{match.group(0)}", text)

    origion_state = replace_numbers_with_underscore(origion_state)
    target_state = replace_numbers_with_underscore(target_state)

    latex = "$\\mathrm{{{} \\: \\rightarrow \\: {}}}$".format(origion_state,target_state )
    return latex