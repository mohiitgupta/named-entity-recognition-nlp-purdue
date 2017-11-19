import subprocess
import argparse
import sys
import gzip
import cPickle

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

class Classifier(object):
    def __init__(self):
        pass

    def train(self):
        """
        Override this method in your class to implement train
        """
        raise NotImplementedError("Train method not implemented")

    def inference(self):
        """
        Override this method in your class to implement inference
        """
        raise NotImplementedError("Inference method not implemented")



def conlleval(p, g, w, filename='tempfile.txt'):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, ww in zip(sl, sp, sw):
            out += ww + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain precision/recall and F1 score '''
    _conlleval = 'conlleval.pl'

    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall    = float(out[8][:-2])
    f1score   = float(out[10])

    return (precision, recall, f1score)











def make_target(label, size):
    # print "label is ", label
    tensor = torch.zeros(size)
    tensor = tensor.long()
    tensor[label] = 1
    # print "tensor is ", tensor
    return tensor.view(1,-1)
    # return torch.LongTensor([label.tolist()])


class MyNNClassifier(Classifier):
    def __init__(self):
        pass

    def train(self):
        pass

    def inference(self):
        pass


def get_input_vector(word_embeddings, prev_label, word):
    word_feature = autograd.Variable(word_embeddings[word])
    prev_label = autograd.Variable(prev_label)
    input_vector = torch.cat((word_feature.view(1,-1), prev_label.view(1,-1)), 1)
    return input_vector

def viterbi_inference(model, sentence, word_embeddings, tag_embeddings, NUM_LABELS):
        dp = np.zeros((NUM_LABELS, len(sentence)+1))
        back_pointers = np.zeros((NUM_LABELS, len(sentence)))
        dp[0][0] = 1
        for i in range(len(sentence)):
            word_table = np.zeros((NUM_LABELS, NUM_LABELS))
            if i == 0:
                input_vector = get_input_vector(word_embeddings, tag_embeddings[NUM_LABELS], sentence[i])
                probs = model(input_vector)
                probs = probs.data.numpy()
                word_table[:,0] = np.multiply(dp[0][0], probs)
                dp[:,i+1] = word_table[:,0]
                back_pointers[:,i] = 128
                continue
            for j in range(NUM_LABELS):
                input_vector = get_input_vector(word_embeddings, tag_embeddings[j], sentence[i])
                probs = model(input_vector)
                probs = probs.data.numpy()
#                 print " probs ", probs
#                 print "dp array ", dp[:,i]
#                 print np.multiply(dp[:, i], probs)
                word_table[:,j] = np.multiply(dp[:, i], probs)
#             print "word table is ", word_table
            dp[:,i+1] = word_table.max(1)
            for k in range(NUM_LABELS):
                for index, element in enumerate(word_table[k]):
                    if element == dp[k, i+1]:
                        back_pointers[k, i] = index
#                 back_pointers[k, i] = word_table[k].index(dp[k,i+1])

#         print "back_pointers ", back_pointers
#         print "dp matrix ", dp[0]
        output_labels = np.zeros(len(sentence), dtype = np.int)
        label_index = len(sentence) - 1
        max_val = dp[:, len(sentence)].max()
        for index, element in enumerate(dp[:, len(sentence)]):
            if element == max_val:
                output_labels[label_index] = index
#                 print "debug2 ", index
                break
#         print "output labels ", output_labels
        for i in range(len(sentence)-1, 0, -1):
#             print "debug ", output_labels[label_index]
            row = back_pointers[output_labels[label_index], i]
            label_index -= 1
            output_labels[label_index] = row
#         print "output labels ", output_labels
        return output_labels



def greedy_inference(model, sentence, word_embeddings, tag_embeddings, NUM_LABELS):
    output_labels = np.zeros(len(sentence))
    prev_label = tag_embeddings[NUM_LABELS]
    for i, word in enumerate(sentence):
        input_vector =  get_input_vector(word_embeddings, prev_label, word)
        probs = model(input_vector)
        max_val, predicted_label = torch.max(probs, 1)
        predicted_label = predicted_label.data[0]
        prev_label = tag_embeddings[predicted_label]
        output_labels[i] = predicted_label
    return output_labels




class NeuralNet(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_input_nodes, num_hidden_nodes, output_dimension):
        super(NeuralNet, self).__init__()
        self.input_linear = nn.Linear(num_input_nodes, num_hidden_nodes)
        self.middle_linear = nn.ReLU()
        self.output_linear = nn.Linear(num_hidden_nodes, output_dimension)

    def forward(self, input_vector):
        out = self.input_linear(input_vector)
        h_relu = self.middle_linear(out)
        y_pred = self.output_linear(h_relu)
        # return F.log_softmax(y_pred)
        return y_pred

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", type=str, default="atis.small.pkl.gz", help="The zipped dataset")

    parsed_args = argparser.parse_args(sys.argv[1:])

    filename = parsed_args.data
    f = gzip.open(filename,'rb')
    train_set, valid_set, test_set, dicts = cPickle.load(f)

    # print "train_set ", train_set

    train_lex, _, train_y = train_set
    valid_lex, _, valid_y = valid_set
    test_lex,  _,  test_y  = test_set

    # print "train_lex ", train_lex
    # print "train_y ", train_y

    idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())

    '''
    To have a look what the original data look like, commnet them before your submission
    '''
    print "length train data ", len(train_lex), " ", len(train_y)
    # print "test lex ", test_lex[0]
    # print "word dictionary is ", idx2word
    # print "label dictionary is ", idx2label
    # print train_lex[0], map(lambda t: idx2word[t], train_lex[0])
    # print train_y[0], map(lambda t: idx2label[t], train_y[0])
    # print test_lex[0], map(lambda t: idx2word[t], test_lex[0])
    # print test_y[0], map(lambda t: idx2label[t], test_y[0])

    '''
    implement you training loop here
    '''
    NUM_LABELS = len(idx2label)
    VOCAB_SIZE = len(idx2word)
    HIDDEN_NODES = 150
    # word_embeddings = torch.rand(VOCAB_SIZE, 300)
    word_embeddings = torch.eye(VOCAB_SIZE, VOCAB_SIZE)
    # tag_embeddings = torch.rand(NUM_LABELS+1, 100)
    tag_embeddings = torch.eye(NUM_LABELS+1, NUM_LABELS+1)
    NUM_INPUT_NODES = len(word_embeddings[0]) + len(tag_embeddings[0])
    # print "number of input nodes ", NUM_INPUT_NODES
    # print "word_embeddings ", word_embeddings
    # print "tag_embeddings ", tag_embeddings
    # input dimension for neural network is concatenation of word and tag tensors
    model = NeuralNet(NUM_INPUT_NODES, HIDDEN_NODES, NUM_LABELS)

    # loss_function = nn.MSELoss(size_average=False)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.3)
    print "using loss function ", loss_function, " and optimizer ", optimizer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    # optimizer = optim.Adam(model.parameters(), lr=0.03)

    for epoch in range(3):
        epoch_loss = 0
        for sentence, labels in zip(train_lex, train_y):
    #         flag = bool(random.getrandbits(1))
    #         if flag:
    #             continue
            prev_label = tag_embeddings[NUM_LABELS]
            for word, label in zip(sentence, labels):
                model.zero_grad()
                word_feature = autograd.Variable(word_embeddings[word])
                prev_label = autograd.Variable(prev_label)
                input_vector = torch.cat((word_feature.view(1,-1), prev_label.view(1,-1)), 1)

                prev_label = tag_embeddings[label]
                # input_vector = autograd.Variable(concat_vec)
                # print "input vector ", input_vector
                label_tensor = torch.LongTensor([label.item()])
                target = autograd.Variable(label_tensor)
    #             target = autograd.Variable(make_target(label, NUM_LABELS))
                probs = model(input_vector)
    #             print "probs ", log_probs
    #             print "target ", target
                loss = loss_function(probs, target)
                epoch_loss += loss.data[0]
    #             print "loss ", loss 
                loss.backward()
                optimizer.step()
        print "epoch number ", epoch, " epoch_loss ", epoch_loss

    # viterbi_inference(test_lex, model, word_embeddings, tag_embeddings, NUM_LABELS)




    '''
    how to get f1 score using my functions, you can use it in the validation and training as well
    '''
    predictions_test = [ map(lambda t: idx2label[t], 
                             viterbi_inference(model, x, word_embeddings, tag_embeddings, NUM_LABELS)) 
                        for x in test_lex ]
    groundtruth_test = [ map(lambda t: idx2label[t], y) for y in test_y ]
    words_test = [ map(lambda t: idx2word[t], w) for w in test_lex ]
    test_precision, test_recall, test_f1score = conlleval(predictions_test, groundtruth_test, words_test)

    print test_precision, test_recall, test_f1score



if __name__ == '__main__':
    main()
