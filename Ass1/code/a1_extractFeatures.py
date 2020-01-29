import numpy as np
import argparse
import json
import csv

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}
PUNCTUATION = {
    '$', '#', '"', '"', '(', ')', ',', '.', ':', 'XX', '``', "''", "-LRB-",
    '-RRB-', 'ADD', 'AFX', 'HYPH'
}

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    # TODO: Extract features that rely on capitalization.
    rtv = np.zeros((173,))
    print(json.dumps(comment, indent=4))
    # 1. Count caps
    for item in comment["originalBody"].split():
        if not item.islower():
            rtv[0] = rtv[0] + 1

    # 2,3,4. Count I, you, he
    tokens = comment["body"].split()

    # future tense verb, if it is in the form of <going to Verb>
    punc_counter = 0
    for token_prev, token_after in zip(tokens[0:-1], tokens[1:]):
        word_prev = token_prev.split("/")[0].lower()
        prop_prev = token_prev.split("/")[1]
        word_after = token_after.split("/")[0].lower()
        prop_after = token_after.split("/")[1]
        # see if there is a going to, this will help detect future tense
        if word_prev == "go" and prop_prev == "VBG" and word_after == "to":
            rtv[6] = rtv[6] + 1

        # this helps detect multi-character punctuation that has 2 or more tokens
        if prop_prev in PUNCTUATION and prop_after in PUNCTUATION and punc_counter == 0:
            punc_counter = 2
        elif punc_counter == -1:
            rtv[7] = rtv[7] + 1
        elif punc_counter >= 2:
            if prop_after in PUNCTUATION:
                punc_counter = punc_counter + 1
            elif punc_counter >= 2:
                punc_counter = -1
    word_count = 0
    for token in tokens:
        word = token.split("/")[0].lower()
        prop = token.split("/")[1]
        if word in SECOND_PERSON_PRONOUNS:
            rtv[2] = rtv[2] + 1
        if prop == "PRP":
            if word in FIRST_PERSON_PRONOUNS:
                rtv[1] = rtv[1] + 1
            elif word in THIRD_PERSON_PRONOUNS:
                rtv[3] = rtv[3] + 1
        # 5, Cordinating Conjunction
        elif prop == "CC":
            rtv[4] = rtv[4] + 1
        # past tense verb
        elif prop == "VBD":
            rtv[5] = rtv[5] + 1
        # future tense. Testing shows that spacy can make sure a "will" is recognized as a noun
        elif (word == "will" and prop == "MD") or word == "shall":
            rtv[6] = rtv[6] + 1
        elif word == "," and prop == ",":
            rtv[7] = rtv[7] + 1
        # this one is for recognized long punctuation
        elif prop == "." and len(word) >= 3:
            rtv[8] = rtv[8] + 1
        elif "NN" == prop or "NNS" == prop:
            rtv[9] = rtv[9] + 1
        elif "NNP" == prop or "NNPS" == prop:
            rtv[10] = rtv[10] + 1
        elif "RB" == prop or "RBS" == prop or "RBR" == prop:
            rtv[11] = rtv[11] + 1
        elif prop == "WDT" or prop == "WP" or prop == "WP$" or prop == "WRB":
            rtv[12] = rtv[12] + 1
        elif word in SLANG:
            rtv[13] = rtv[13] + 1
        # count average token length
        if not prop in PUNCTUATION:
            rtv[15] = rtv[15] + len(word)
            word_count = word_count + 1

        # counte average sentence length
    for sentence in comment["body"].split("\n"):
        rtv[14] = rtv[14] + len(sentence.split())
    rtv[14] = rtv[14]/len(comment["body"].split("\n"))
    # calculate avg
    rtv[15] = rtv[15]/word_count
    rtv[16] = len(comment["body"].split("\n"))
    print(rtv)
    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    # TODO: Extract features that do not rely on capitalization.
    # print('TODO')
    return rtv
    
    
def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment
    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''    
    print('TODO')


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    bristo_norm = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
    bristo_norm = "./BristolNorms+GilhoolyLogie.csv"
    bristo_norm_dict = {}
    with open(bristo_norm, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',',)
        thing = False
        for row in reader:
            if thing:
                entry = []
                entry.append(float(row[3]))
                entry.append(float(row[4]))
                entry.append(float(row[5]))
                bristo_norm_dict[row[1]] = entry
            else:
                thing = True

    # TODO: Use extract1 to find the first 29 features for each
    extract1(data[6])
    print()
    # data point. Add these to feats.
    # TODO: Use extract2 to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    print('TODO')

    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True, default="preproc.json")
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        
    # python3.6 a1_extractFeatures.py -i preproc.json -o feats.npz -p "/."
    main(args)

