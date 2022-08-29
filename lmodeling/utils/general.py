import numpy as np
import torch
import sys
sys.path.insert(1, '../')
from data_utils import get_lm_corpus

def index_padder(corpus, data, dataset):
    tlen, bz = data.size()
    lg = np.array(data.view(1,-1).detach().cpu())[0]
    new_list, indexes = [], []
    
    if dataset == 'wt103':
        forbid_nm = [corpus.vocab.sym2idx[str(inx)] for inx in range(2,100)]
        symb = ['<eos>','<unk>','~','`','!','#','@','@-@','@,@','$','%','^','&','*','(',')','-','_','+','=',':',';','.',',','?','<','>',"\\",'\"','\'','{','}','|','[', ']']
    elif dataset == 'ptb':
        symb = ['<eos>','<unk>','#','$','&','\'']

    chars = [i for i in 'abcdefiknrsuxz']
    forbid_sym = [corpus.vocab.sym2idx[str(inx)] for inx in symb]
    chars2 = ['a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.', 'i.', 'j.', 'l.', 'm.', 'n.', 'o.', 'p.', 'r.', 's.', 't.', 'v.']
    forbid_char = [corpus.vocab.sym2idx[str(inx)] for inx in chars]
    forbid_char2 = [corpus.vocab.sym2idx[str(inx)] for inx in chars2]
    if dataset == 'wt103':
        forbidden = forbid_nm + forbid_sym + forbid_char + forbid_char2
    elif dataset == 'ptb':
        forbidden = forbid_sym + forbid_char + forbid_char2
    
    padded = np.in1d(lg,np.array(forbidden)).reshape(data.size())
    padded = ~padded
    return torch.tensor(padded*1)

    #return torch.tensor(np.array(new_list).reshape(tlen,bz))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')
    parser.add_argument('--datadir', type=str, default='../../data/wiki-103',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='ptb',
                        choices=['ptb', 'wt2', 'wt103', 'lm1b', 'enwik8', 'text8'],
                        help='dataset name')
    args = parser.parse_args()

    corpus = get_lm_corpus(args.datadir, args.dataset)
    #print('Vocab size : {}'.format(len(corpus.vocab.idx2sym)))
    #print(corpus.vocab.sym2idx['3'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = np.random.randint(0,12333,size=(80,60))
    data= torch.tensor(a).to(device)
    print(data.size())
    print(index_padder(corpus, data))
