import os, math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import time
from tensorflow import keras
from tensorflow.python.keras import layers
from miniMazeTest import MazeGeneration, __init__
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.input_spec import InputSpec

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--targets', action='store_true', help='Use target space')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--nhid', type=int, default=16, help="num hidden units per layer")
parser.add_argument('--nhl', type=int, default=1, help="num hidden layers")
parser.add_argument('--nel', type=int, default=3, help="num exit layers")
parser.add_argument('--lr', type=float, default=0.001,help='learning_rate')
parser.add_argument('--max_epoch', type=int, default=600,help='max_epoch')
parser.add_argument('--af', type=str, default="tanh")
parser.add_argument('--recurrent', action='store_true')
parser.add_argument('--extra_adj_layer', action='store_true')
parser.add_argument('--log_results', action='store_true', help='Log results in csv format')
parser.add_argument('--sequence_length', type=int, default=6, help='sequence_length')
parser.add_argument('--dataset_size', type=int, default=None, help='training set size')
parser.add_argument('--mbs', type=int, default=512)
parser.add_argument('--problem', type=str, default="parity", help='parity or bipartite or parity_all_odd or cycle_lengths_parity')

args = parser.parse_args()

#To-do: Ensure that the parity stuff is changed to fit maze generation.
if args.problem == "MazeGeneration":
    assert args.nel == 1
    args.max_epoch = min(6000, args.max_epoch)
    if args.targets:
        assert args.lr == 0.001
        args.max_epoch = min(2000, args.max_epoch)

if args.dataset_size == None:
    if args.sequence_length <= 20:
        args.dataset_size = 10000
    else:
        args.dataset_size = 1000000

if args.nhl < 0:
    args.nhl = (args.sequence_length // 2) + 1

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')

def calculateInputOutputSequences(seqLength, self, seed1):
    if args.problem == "MazeGeneration":
        np.random.seed(seed1)
        inputs, labels = MazeGeneration.generate(self, labeller_function = is_maze_solvable)
        assert len(labels.shape) == 2
        assert labels.shape[1] == seqLength
        labels = np.expand_dims(labels, 2)
        return inputs, labels
    else:
        raise Exception("Unknown problem " + args.problem)


seq_length = args.sequence_length
dataset_size = min(args.dataset_size, math.factorial(seq_length))
train_inputs, train_labels = calculateInputOutputSequences(seq_length, dataset_size, seed1 = 1)
if args.problem != "MazeGeneration":
    train_labels = train_labels.reshape((-1))

n = (int)(len(train_inputs) * 0.8)
test_inputs, test_labels = train_inputs[n:], train_labels[n:]
train_inputs, train_labels = train_inputs[:n], train_labels[:n]
test_base_rate = test_labels.mean()
train_base_rate = train_labels.mean()
if not args.log_results:
    print()
    print("train data shape, input", train_inputs.shape,"output", train_labels.shape)
    print("test data shape, input", test_inputs.shape, "output", test_labels.shape)
    print()

fast_adjacency_matrix_multiply = False
if fast_adjacency_matrix_multiply:
    test_inputs = np.argmax(test_inputs, axis=2).astype(np.int32)
    train_inputs = np.argmax(train_inputs, axis= 2).astype(np.int32)
    
nhid = args.nhid
learning_rate = args.lr
dropout_rate = args.dropout
num_epochs = args.max_epoch
adj_normalised = True
use_binary_output = True

class MazeGCN(tf.keras.Model):
    def __init__(self, nhid, nclass, num_hidden_layers, num_exit_layers, use_target_space,recurrent, seq_length, with_scu_rnn_projection, realisation_batch_size, is_node_prediction_task, af, gcn2alpha=0, gcn2lambda=None):
        super(MazeGCN, self).__init__()
        self.mylayers=[]
        self.seq_length = seq_length
        self.gcn2alpha = gcn2alpha
        self.gcn2lambda = gcn2lambda
        for l in range(num_hidden_layers):
            self.mylayers.append("ADJ")
            if self.gcn2lambda != None:
                self.mylayers.append("CacheForAddLater")
            if l == 0 or not recurrent:
                new_layer = layers.Dense(units=nhid, activation = None)
            self.mylayers.append(new_layer)
            if self.gcn2lambda != None:
                self.mylayers.append(["Add", math.log(1 + self.gcn2lambda/ (1+1))])
            self.mylayers.append(layers.Activation(af))
            if self.gcn2alpha > 0:
                self.mylayers.append("GCN2ShortcutLayer")

        if args.extra_adj_layer:
            self.mylayers.append("ADJ")
            if self.gcn2alpha > 0:
                self.mylayers.append("GCN2ShortcutLayer")

        assert nclass == 2
        if use_binary_output:
            output_activation_function = "sigmoid"
            nclass = 1
        else:
            output_activation_function = None
        if not is_node_prediction_task:
            self.mylayers.append(layers.Flatten())

            for l in range(num_exit_layers - 1):
                if use_target_space:
                    self.mylayers.append(tf.TSDense(nhid, realisation_batch_size=realisation_batch_size, activation=af,pseudoinverse_l2_regularisation=args.prc))
                else:
                    self.mylayers.appendlaters.Dense(units = nhid, activation = af)
            self.mylayers.append(layers.Dense(units = nclass, activation = output_activation_function))
        else:
            new_layer = layers.Dense(units = 1, activation = output_activation_function)
            self.mylayers.append(new_layer)

    def fast_matmul_adjacency_matrix(self, adjacency_matrix, x):
        assert len(x.shape) == 3
        if fast_adjacency_matrix_multiply:
            assert len(adjacency_matrix.shape) == 2
            result = tf.gather(x, adjacency_matrix, axis = 1, batch_dims = 1)
        else:
            assert len(adjacency_matrix.shape) == 3
            result = tf.matmul(adjacency_matrix, x)
        return result
            
    def call(self, adjacency_matrix_batch, training):
        x0 = tf.fill(dims=[tf.shape(adjacency_matrix_batch)[0], self.seq_length, 1], value = 0.0) + tf.eye(self.seq_length, dtype = tf.float32)
        x = x0 + 0

        prev_add_state = None
        for L in self.mylayers:
            if "List" in str(type(L)):
                [l, arg] = L
            else:
                l = L
                arg = None
            if isinstance(l, str) and l == "ADJ":
                x += self.fast_matmul_adjacency_matrix(adjacency_matrix_batch, x)
                if adj_normalised:
                    x = x * 0.5
            elif isinstance(l, str) and l == "Add":
                assert arg!= None
                x = x * arg+prev_add_state[0]*(l - arg)
            elif isinstance(l, str) and l == "CacheForAddLater":
                prev_add_state = [x]
            else:
                x = l(x)
        return x

gnn_model = MazeGCN(nhid, nclass = 2, num_hidden_layers = args.nhl, num_exit_layers = args.nel, use_target_space = False, recurrent = args.recurrent, seq_length = seq_length, with_scu_rnn_projection = False, realisation_batch_size = 0, is_node_prediction_task = (args.problem == "cycle_lengths_parity"), gcn2alpha = 0, gcn2lambda = 0, af = args.af)

if use_binary_output:
    loss_function = keras.losses.BinaryCrossentropy(from_logits = False)
    accuracy_metric = keras.metrics.BinaryAccuracy(name = "acc")
else:
    loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    accuracy_metric = keras.metrics.SparseCategoricalAccuracy(name = "acc")

other_metrics = [keras.metrics.TruePositives(), keras.metrics.FalsePositives(), keras.metrics.TrueNegatives(), keras.metrics.FalseNegatives()]
optimizer = optimizer = keras.optimizers.Adam(learning_rate)
gnn_model.compile(
    optimizer = optimizer,
    loss = loss_function,
    metrics = [accuracy_metric] + other_metrics,
)

start_time = time.time()
if args.log_results:
    print ("problem, seq_length, epoch, train_set_size, test_set_size, learning_rate, nhids, num_gcn_layers, num_exit_layers, mbs, val_acc, acc, val_loss, loss, layer_loss, prc, cpu_time, test_base_rate, train_base_rate, true_positives, false_positives, true_negatives, false_negatives, extra_adj_layer, adj_normalised, use_binary_output, af")
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs = None):
            keys = list(logs.keys())
            if "val_acc" in keys:
                layer_loss = 0
                print(args.problem, seq_length, epoch + 1, len(train_inputs), len(test_inputs), learning_rate, args.nhid, args.nhl, args.nel, args.mbs, logs["val_acc"], logs["acc"], logs["val_loss"], logs["loss"], layer_loss, time.time()-start_time, test_base_rate, train_base_rate, logs["val_true_positives"], logs["val_false_positives"], logs["val_true_negatives"], logs["val_false_negatives"], args.extra_adj_layer, adj_normalised,use_binary_output, args.af, sep=",")
    callbacks = CustomCallback()
    verbose = 0
else:
    callbacks = None
    verbose = 1

history = gnn_model.fit(
    x = train_inputs,
    y = train_labels,
    epochs = num_epochs,
    batch_size = args.mbs,
    validation_data = (test_inputs, test_labels),
    validation_freq = 10,
    verbose = verbose, callbacks = callbacks
)
