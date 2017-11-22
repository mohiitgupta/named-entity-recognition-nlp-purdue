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

def create_embedding_matrix(word_embeddings, tag_embeddings, train_lex, train_y, NUM_LABELS, VOCAB_SIZE):
    word_embedding_list = []
    label_list = []
    
    for sentence, labels in zip(train_lex, train_y):
        prev_label = tag_embeddings[NUM_LABELS]
        prev_word = word_embeddings[VOCAB_SIZE]
        for word, label in zip(sentence, labels):
            word_embedding = word_embeddings[word]
            input_vector = torch.cat((prev_word.view(1,-1), word_embedding.view(1,-1), prev_label.view(1,-1)), 1)
            word_embedding_list.append(input_vector)
            prev_label = tag_embeddings[label]
            prev_word = word_embeddings[word]
            # for mse loss
            label_tensor = torch.LongTensor(NUM_LABELS).zero_().view(1,-1)
            label_tensor[0,label] = 1
            label_tensor = label_tensor.float()
            label_list.append(label_tensor)
            
            # for cross entropy loss since multi target not supported
            # label_tensor = torch.LongTensor([label.item()])
            # label_tensor = label_tensor.long()
            # label_list.append(label_tensor)
    print "word embedding list ", len(word_embedding_list)
    print "label list ", len(label_list)
    return word_embedding_list, label_list

def get_input_vector(word_embeddings, prev_label, word, prev_word):
    word_embedding = word_embeddings[word]
    word_feature = autograd.Variable(word_embeddings[word])
    prev_label = autograd.Variable(prev_label)
    prev_word = autograd.Variable(prev_word)
    input_vector = torch.cat((prev_word.view(1,-1), word_feature.view(1,-1), prev_label.view(1,-1)), 1)
    # print "input vector ", input_vector
    return input_vector

class NeuralNet(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_input_nodes, num_hidden_nodes, output_dimension):
        super(NeuralNet, self).__init__()
        self.input_linear = nn.Linear(num_input_nodes, num_hidden_nodes)
        self.output_linear = nn.Linear(num_hidden_nodes, output_dimension)

    def forward(self, input_vector):
        out = self.input_linear(input_vector)
        out = F.tanh(out)
        out = self.output_linear(out)
        out = F.softmax(out)
        return out

class MyNNClassifier(Classifier):
    def __init__(self):
        pass

    def train(self, word_embeddings, tag_embeddings, train_lex, train_y, NUM_LABELS, VOCAB_SIZE):
        word_embedding_list, label_list = create_embedding_matrix(word_embeddings,
         tag_embeddings, train_lex, train_y, NUM_LABELS, VOCAB_SIZE)
        HIDDEN_NODES = 1000
        word_embedding_list = torch.stack(word_embedding_list)
        word_embedding_list = torch.squeeze(word_embedding_list)
        label_list = torch.stack(label_list)
        label_list = torch.squeeze(label_list)
        NUM_INPUT_NODES = word_embedding_list[0].size()[0]
        print "number of input nodes ", NUM_INPUT_NODES
        print "word_embeddings ", word_embedding_list.size()
        print "label list ", label_list.size()

        model = NeuralNet(NUM_INPUT_NODES, HIDDEN_NODES, NUM_LABELS)
        loss_function = nn.MSELoss()
        # loss_function = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.01
        # #                       , momentum=0.9
        #                      )
        words = autograd.Variable(word_embedding_list)
        label = autograd.Variable(label_list
                                  , requires_grad=False
                         )
        optimizer = optim.Adam(model.parameters(), lr=0.0005
            # , weight_decay=0.00001
            )
        for epoch in range(1202):
            probs = model(words)
            loss = loss_function(probs, label)
            print "loss ", loss.data[0], " epoch ", epoch
            if epoch % 200 == 1:
                torch.save(model.state_dict(), 'parameters_' + str(epoch) + '.pt')
                torch.save(word_embeddings, 'word_embeddings.pt')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model

    def inference(self):
        pass

    def greedy_inference(self, model, sentence, word_embeddings, tag_embeddings, NUM_LABELS, VOCAB_SIZE):
        output_labels = np.zeros(len(sentence))
        prev_label = tag_embeddings[NUM_LABELS]
        prev_word = word_embeddings[VOCAB_SIZE]
        # prob_list = []
        for i, word in enumerate(sentence):
            input_vector =  get_input_vector(word_embeddings, prev_label, word, prev_word)
            probs = model(input_vector)
            # prob_list.append(probs)
            max_val, predicted_label = torch.max(probs, 1)
            predicted_label = predicted_label.data[0]
            prev_label = tag_embeddings[predicted_label]
            output_labels[i] = predicted_label
            prev_word = word_embeddings[word]
        # print output_labels
        return output_labels

    def viterbi_inference(self, model, sentence, word_embeddings, tag_embeddings, NUM_LABELS, VOCAB_SIZE):
        dp = np.zeros((NUM_LABELS, len(sentence)+1))
        back_pointers = np.zeros((NUM_LABELS, len(sentence)))
        dp[0][0] = 1
        prev_word = word_embeddings[VOCAB_SIZE]
        for i in range(len(sentence)):
            word_table = np.zeros((NUM_LABELS, NUM_LABELS))
            if i == 0:
                input_vector = get_input_vector(word_embeddings, tag_embeddings[NUM_LABELS], sentence[i], prev_word)
                probs = model(input_vector)
                probs = probs.data.numpy()
                word_table[:,0] = np.multiply(dp[0, 0], probs)
                dp[:,i+1] = word_table[:,0]
                back_pointers[:,i] = 128
                prev_word = word_embeddings[sentence[i]]
                continue
            for j in range(NUM_LABELS):
                input_vector = get_input_vector(word_embeddings, tag_embeddings[j], sentence[i], prev_word)
                probs = model(input_vector)
                probs = probs.data.numpy()
                word_table[:,j] = np.multiply(dp[j, i], probs)
            prev_word = word_embeddings[sentence[i]]
            dp[:,i+1] = word_table.max(1)
            for k in range(NUM_LABELS):
                for index, element in enumerate(word_table[k]):
                    if element == dp[k, i+1]:
                        back_pointers[k, i] = index
        output_labels = np.zeros(len(sentence), dtype = np.int)
        label_index = len(sentence) - 1
        max_val = dp[:, len(sentence)].max()
        for index, element in enumerate(dp[:, len(sentence)]):
            if element == max_val:
                output_labels[label_index] = index
                break
        for i in range(len(sentence)-1, 0, -1): #for example 18 to 1
            row = back_pointers[output_labels[label_index], i]
            label_index -= 1
            output_labels[label_index] = row
        return output_labels

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", type=str, default="atis.small.pkl.gz", help="The zipped dataset")

    parsed_args = argparser.parse_args(sys.argv[1:])

    filename = parsed_args.data
    f = gzip.open(filename,'rb')
    train_set, valid_set, test_set, dicts = cPickle.load(f)

    train_lex, _, train_y = train_set
    valid_lex, _, valid_y = valid_set
    test_lex,  _,  test_y  = test_set

    idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())

    VOCAB_SIZE = len(idx2word)
    NUM_LABELS = len(idx2label)
    '''
    To have a look what the original data look like, commnet them before your submission
    '''
    #initialize myNNClassifier object
    myNNClassifier = MyNNClassifier()

    '''
    implement you training loop here
    #Note : Takes around 1 hour to train the model with 1200 epochs and final MSE loss of around 0.0002
    '''
    # word_embeddings = torch.rand(VOCAB_SIZE+1, 300)
    word_embeddings = torch.load('word_embeddings.pt')
    #     word_embeddings = torch.eye(VOCAB_SIZE, VOCAB_SIZE)
    tag_embeddings = torch.eye(NUM_LABELS+1, NUM_LABELS+1)
    model = myNNClassifier.train(word_embeddings, tag_embeddings, train_lex, train_y, NUM_LABELS, VOCAB_SIZE)

    #for using pretrained model
    # word_embeddings = torch.load('word_embeddings.pt')
    # # word_embeddings = torch.eye(VOCAB_SIZE, VOCAB_SIZE)
    # tag_embeddings = torch.eye(NUM_LABELS+1, NUM_LABELS+1)
    # model = NeuralNet(728, 1000, NUM_LABELS) 
    # model.load_state_dict(torch.load('parameters_1.pt'))
    # myNNClassifier.train(model, word_embeddings, tag_embeddings, train_lex, train_y, NUM_LABELS, VOCAB_SIZE)
    # print "model ", model.state_dict()

    '''
    how to get f1 score using my functions, you can use it in the validation and training as well
    '''
    # predictions_train = [ map(lambda t: idx2label[t],
    #  myNNClassifier.greedy_inference(model, x, word_embeddings, tag_embeddings, NUM_LABELS, VOCAB_SIZE)) for x in train_lex]
    # # print "predictions ", predictions_train[0]
    # groundtruth_train = [ map(lambda t: idx2label[t], y) for y in train_y ]
    # # print "groundtruth ", groundtruth_train[0]
    # words_train = [ map(lambda t: idx2word[t], w) for w in train_lex ]
    # train_precision, train_recall, train_f1score = conlleval(predictions_train, groundtruth_train, words_train)
    # print "Training results ", train_precision, train_recall, train_f1score


    # predictions_valid = [ map(lambda t: idx2label[t],
    #  myNNClassifier.greedy_inference(model, x, word_embeddings, tag_embeddings, NUM_LABELS, VOCAB_SIZE)) for x in valid_lex]
    # # print "predictions ", predictions_valid[0]
    # groundtruth_valid = [ map(lambda t: idx2label[t], y) for y in valid_y ]
    # # print "groundtruth ", groundtruth_valid[0]
    # words_valid = [ map(lambda t: idx2word[t], w) for w in valid_lex ]
    # valid_precision, valid_recall, valid_f1score = conlleval(predictions_valid, groundtruth_valid, words_valid)
    # print "Validation results ", valid_precision, valid_recall, valid_f1score

    predictions_test = [ map(lambda t: idx2label[t],
     myNNClassifier.greedy_inference(model, x, word_embeddings, tag_embeddings, NUM_LABELS, VOCAB_SIZE)) for x in test_lex]
    # print "predictions ", predictions_test[0]
    groundtruth_test = [ map(lambda t: idx2label[t], y) for y in test_y ]
    # print "groundtruth ", groundtruth_test[0]
    words_test = [ map(lambda t: idx2word[t], w) for w in test_lex ]
    test_precision, test_recall, test_f1score = conlleval(predictions_test, groundtruth_test, words_test)
    print "Test Results ", test_precision, test_recall, test_f1score


if __name__ == '__main__':
    main()
