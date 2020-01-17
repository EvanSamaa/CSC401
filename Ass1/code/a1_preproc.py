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


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)
            data = json.load(open(fullFile))
            # TODO: select appropriate args.max lines
            start_index = args.ID[0] % len(data)
            end_index = start_index + args.max
            sub_data = []
            if end_index < len(data):
                sub_data = data[start_index:end_index]
            else:
                circular_end = end_index - len(data)
                sub_data = data[start_index:]
                sub_data = subdata + data[0:circular_end]
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            for comment in sub_data:
                comment_d = json.loads(comment)
                processed_data = {}
                processed_data["id"] = comment_d["id"]
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
                processed_data["cat"] = fullFile.split("/")[-1]
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
                processed_data["body"] = preproc1(comment_d["body"], [1, 2, 3, 4, 5])
            # TODO: append the result to 'allOutput'
                allOutput.append(processed_data)
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", type=int, default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)
