import random,json,math

def read_fasta(filename, keep_ids=False):
    with open(filename) as filehandle:
        raw_data = filehandle.readlines()
    seqs = []
    seq=''
    seq_id=''
    for line in raw_data:
        if line [0] == '>':
            if len(seq)>0:
                seqs.append(seq if not keep_ids else (seq, seq_id))
                seq=''
            seq_id = line[1:].strip()
        else:
            seq += line.strip()

    seqs.append(seq if not keep_ids else (seq, seq_id)) # make sure we include the final sequence
    return seqs


seq_classes = {'cyto':0, 'mito':1, 'nucleus':2, 'secreted':3}

# _ represents padding, X represents unknown
seq_tokens=['_']+["G","P","A","V","L","I","M","C","F","Y","W","H","K","R","Q","N","E","D","S","T"] +['X']+["U","O"] +["B","J","Z"]

# This is slow! Use it once when you get a new dataset
def check_tokens(data):
    for seq,c in data:
        for char in seq:
            if char not in seq_tokens:
                print('Unknown token!', char)


def load_data():

    labelled_data = []
    for c in seq_classes.keys():
        labelled_data.extend([(seq,seq_classes[c]) for seq in read_fasta('./data/'+c+'.fasta')])

    test_seqs = read_fasta('./data/blind.fasta', keep_ids=True)

    return labelled_data, test_seqs

def generate_split_dataset(test_frac=0.1):
    all_train_data, eval_data = load_data()
    random.shuffle(all_train_data)
    test_ix = math.floor(len(all_train_data)*(1-test_frac))
    train_data = all_train_data[:test_ix]
    test_data = all_train_data[test_ix:]
    # with open('./data/train.json','w') as fp:
    #     json.dump(train_data, fp)
    # with open('./data/test.json','w') as fp:
    #     json.dump(test_data, fp)
    with open('./data/blind.json','w') as fp:
        json.dump(eval_data, fp)

if __name__ == '__main__':
    generate_split_dataset()
