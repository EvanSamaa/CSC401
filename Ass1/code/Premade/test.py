import sys
import argparse
import os
import json
import re
import spacy


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

def preproc1(comment , steps=range(1, 5)):
    ''' This function pre-processes a single comment
    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''
    modComm = comment
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        # modComm = html.escape(modComm, quote=True)
        modComm = str(modComm.encode('ascii', 'xmlcharrefreplace')).encode('utf-8').decode('unicode_escape')[2:-1]
    if 3 in steps:  # remove URLs
        modComm = re.sub(r"(http|http|www).*\s", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        modCom_list = modComm.split(" ")
        for item in modCom_list:
            if item == " " or item == "":
                modCom_list.remove(item)
        modComm = ""
        for word in modCom_list:
            modComm = modComm + word + " "
    # TODO: get Spacy document for modComm
    utt = nlp(modComm)
    modCom_list = []
    # TODO: use Spacy document for modComm to create a string.
    # adding tag and changing to lemma
    for sent in utt.sents:
        for token in sent:
            # print(token)
            lemma = token.lemma_
            if token.lemma_[0] == '-':
                lemma = token.text
            word = lemma + "/" + token.tag_
            modCom_list.append(word)
        modCom_list.append("\n")
    modComm = ""
    modCom_list = modCom_list[:-1]
    for word in modCom_list:
        if word != "\n":
            modComm = modComm + word + " "
        else:
            modComm = modComm + word
    return modComm
comment = "I'm gonna do some things. I'll do it"
print(preproc1(comment).split("\n"))
A = 0

print(len("asdkjgbakerg"))