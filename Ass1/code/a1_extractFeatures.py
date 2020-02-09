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
AOA_IMG_FAM_DICT = {}
WARRINER_DICT = {}

ALT_DICT = {}
RIGHT_DICT = {}
LEFT_DICT = {}
CENTER_DICT = {}

ALT_NPARR = np.zeros((1,))
RIGHT_NPARR = np.zeros((1,))
LEFT_NPARR = np.zeros((1,))
CENTER_NPARR = np.zeros((1,))

# extract the first 29 features, returns an array
def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    # TODO: Extract features that rely on capitalization.
    rtv = np.zeros((173,))
    # print(json.dumps(comment, indent=4))
    # 1. Count caps
    try:
        for item in comment["originalBody"].split():
            if item.isupper() and len(item) >= 3:
                rtv[0] = rtv[0] + 1
    except:
        for item in comment["body"].split():
            if (item.split("/")[0].isupper()) and len(item.split("/")[0]) >= 3:
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
    aoa_arr = np.array((0,))
    img_arr = np.array((0,))
    fam_arr = np.array((0,))
    v_mean_arr = np.array((0,))
    a_mean_arr = np.array((0,))
    d_mean_arr = np.array((0,))
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
        # get AOA, IMG and FAM
        try:
            scores = AOA_IMG_FAM_DICT[word]
            print(scores)
            aoa_arr = np.append(aoa_arr, scores[0])
            img_arr = np.append(img_arr, scores[1])
            fam_arr = np.append(fam_arr, scores[2])
        except:
            a=2
        # get the other scores
        try:
            scores = WARRINER_DICT[word]
            v_mean_arr = np.append(v_mean_arr, scores[0])
            a_mean_arr = np.append(a_mean_arr, scores[1])
            d_mean_arr = np.append(d_mean_arr, scores[2])
        except:
            a=3
    for sentence in comment["body"].split("\n"):
        print("This is a sent", sentence)
        rtv[14] = rtv[14] + len(sentence.split())
    print("")
    rtv[14] = rtv[14]/len(comment["body"].split("\n"))
    # calculate avg
    rtv[15] = rtv[15]/word_count
    rtv[16] = len(comment["body"].split("\n"))
    if aoa_arr.shape[0] != 1:
        rtv[17] = np.average(aoa_arr[1:])
        rtv[18] = np.average(img_arr[1:])
        rtv[19] = np.average(fam_arr[1:])
        rtv[20] = np.std(aoa_arr[1:])
        rtv[21] = np.std(img_arr[1:])
        rtv[22] = np.std(fam_arr[1:])
    if v_mean_arr.shape[0] != 1:
        rtv[23] = np.average(v_mean_arr[1:])
        rtv[24] = np.average(a_mean_arr[1:])
        rtv[25] = np.average(d_mean_arr[1:])
        rtv[26] = np.std(v_mean_arr[1:])
        rtv[27] = np.std(a_mean_arr[1:])
        rtv[28] = np.std(d_mean_arr[1:])
    # print('TODO')
    # print(rtv)
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
    feat_per_comment = None
    if comment_class == "Left":
        feat_per_comment = LEFT_NPARR[LEFT_DICT[comment_id],:]
    elif comment_class == "Right":
        feat_per_comment = RIGHT_NPARR[RIGHT_DICT[comment_id],:]
    elif comment_class == "Center":
        feat_per_comment = CENTER_NPARR[CENTER_DICT[comment_id], :]
    elif comment_class == "Alt":
        feat_per_comment = ALT_NPARR[ALT_DICT[comment_id],:]
    feats = np.concatenate((feats[0:29], feat_per_comment))
    return feats
# loads the features ((N, 144) arrays) into the programs
def load_feats():
    f = open("/u/cs401/A1/feats/Alt_IDs.txt", "r")
    temp = f.read().split("\n")
    for id,index in zip(temp, range(0,len(temp))):
        ALT_DICT[id] = index
    f.close()
    f = open("/u/cs401/A1/feats/Right_IDs.txt", "r")
    temp = f.read().split("\n")
    for id, index in zip(temp, range(0, len(temp))):
        RIGHT_DICT[id] = index
    f.close()
    f = open("/u/cs401/A1/feats/Left_IDs.txt", "r")
    temp = f.read().split("\n")
    for id, index in zip(temp, range(0, len(temp))):
        LEFT_DICT[id] = index
    f.close()
    f = open("/u/cs401/A1/feats/Center_IDs.txt", "r")
    temp = f.read().split("\n")
    for id, index in zip(temp, range(0, len(temp))):
        CENTER_DICT[id] = index
    f.close()
    global ALT_NPARR
    global RIGHT_NPARR
    global LEFT_NPARR
    global CENTER_NPARR
    ALT_NPARR = np.load("/u/cs401/A1/feats/Alt_feats.dat.npy")
    RIGHT_NPARR = np.load("/u/cs401/A1/feats/Right_feats.dat.npy")
    LEFT_NPARR = np.load("/u/cs401/A1/feats/Left_feats.dat.npy")
    CENTER_NPARR = np.load("/u/cs401/A1/feats/Center_feats.dat.npy")

def load_norms():
    bristo_norm = "/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv"
    bristo_norm = "./Premade/BristolNorms+GilhoolyLogie.csv"
    warriner_norm = "/u/cs401/Wordlists/Ratings Warriner et al.csv"
    warriner_norm = "./Premade/Ratings_Warriner_et_al.csv"
    bristo_norm_dict = {}
    warriner_norm_dict = {}
    with open(bristo_norm, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',',)
        thing = False
        for row in reader:
            if thing:
                try:
                    entry = []
                    entry.append(int(row[3]))
                    entry.append(int(row[4]))
                    entry.append(int(row[5]))
                    bristo_norm_dict[row[1]] = entry
                except:
                    pass
            else:
                thing = True
    AOA_IMG_FAM_DICT = bristo_norm_dict
    with open(warriner_norm, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',',)
        thing = False
        for row in reader:
            if thing:
                try:
                    entry = []
                    entry.append(float(row[2]))
                    entry.append(float(row[5]))
                    entry.append(float(row[8]))
                    warriner_norm_dict[row[1]] = entry
                except:
                    pass
            else:
                thing = True
    # TODO: Use extract1 to find the first 29 features for each
    WARRINER_DICT = warriner_norm
def main(args):

    load_feats()
    load_norms()
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))
    #obtain dictionaries that contains the AOA and Warriner data

    classes = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}
    for i in range (0, len(data)):
        feat_per_comment = extract1(data[i])
        # print(i)
        try:
            feat_per_comment = extract2(feat_per_comment, data[i]["cat"], data[i]["id"])
            # print(feat_per_comment)
        except:
            print("failure to get feature ", data[i])
        feats[i, :] = np.append(feat_per_comment, classes[data[i]["cat"]])
    # data point. Add these to feats.
    # TODO: Use extract2 to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    np.savez_compressed(args.output, feats)
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True, default="preproc.json")
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()
    # python3.7 a1_extractFeatures.py -i ./sample_outputs/sample_out.json -o sample_feats.npz
    #

    sample = None
    mine = None
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    with np.load('sample_feats.npz') as data:
        mine = data['arr_0']
    with np.load("./sample_outputs/sample.npz") as data:
        sample = data['arr_0']
    # print(mine[1])
    # print(sample[1])
    # A[2]
    main(args)

