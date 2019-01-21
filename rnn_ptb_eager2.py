# -*- coding: utf-8 -*-

import os
import numpy as np 
import tensorflow as tf 
from tensorflow.contrib.eager.python import tfe 
from Datasets import Datasets
from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn

layers = tf.keras.layers

class PTBModel(tf.keras.Model):

    def __init__(self, 
                 vocab_size,       # 词表数目，由Dataset类中的数据集决定
                 embedding_dim,    # embedding
                 hidden_dim,       # LSTM层的隐含层神经元个数
                 num_layers,       # LSTM层数
                 sequence_length,  # RNN处理序列的长度
                 keep_prob):       # 在训练时加入dropout. 0.8

        super(PTBModel, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.keep_prob = keep_prob
        self.vocab_size = vocab_size

        # Embedding层
        self.embedding = layers.Embedding(vocab_size,
                    embedding_dim, input_length=sequence_length)

        # 这里RNN没有调用keras里面的RNN，便于对记忆的操作
        # tf.contrib.checkpoint.List用于存储cell变量，此处不宜用list直接存储
        self.cells = tf.contrib.checkpoint.List([
            tf.nn.rnn_cell.LSTMCell(num_units=hidden_dim)
            for _ in range(num_layers)])

        # Linear
        self.linear = layers.Dense(vocab_size,
            kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1))

    def call(self, input_seq, training):
        # input_seq.shape = (batch_size, sequence_length)
        batch_size = int(input_seq.shape[0])
        assert input_seq.shape == (batch_size, self.sequence_length)
        # embedding
        input_seq = self.embedding(input_seq)
        assert input_seq.shape == (batch_size, self.sequence_length, self.embedding_dim)
        # Dropout
        if training:
            input_seq = tf.nn.dropout(input_seq, keep_prob=self.keep_prob)

        # RNN
        for c in self.cells:
            state = c.zero_state(batch_size, tf.float32)
            outputs = []
            input_seq = tf.unstack(input_seq, num=int(input_seq.shape[1]), axis=1)
            for inp in input_seq:
                output, state = c(inp, state)
                outputs.append(output)

            input_seq = tf.stack(outputs, axis=1)
            if training:
                input_seq = tf.nn.dropout(input_seq, keep_prob=self.keep_prob)
        assert input_seq.shape == (batch_size, self.sequence_length, self.hidden_dim)
        # 全连接层
        output_seq = tf.reshape(input_seq, [-1, self.hidden_dim])
        output_seq = self.linear(output_seq)
        assert output_seq.shape == (batch_size*self.sequence_length, self.vocab_size)
        return output_seq


def train(model, optimizer, train_data, sequence_length, clip_ratio):
    train_losses = []
    # range把原来的sequence截成多部分，分别训练
    for batch, i in enumerate(range(0, train_data.shape[1]-sequence_length, sequence_length)):
        # 获取数据
        train_seq = tf.convert_to_tensor(train_data[:, i:i + sequence_length])
        train_target = tf.convert_to_tensor(train_data[:, i + 1:i + 1 + sequence_length])
        assert train_seq.shape[1] == train_target.shape[1] == sequence_length

        with tfe.GradientTape() as tape:
            labels = tf.reshape(train_target, [-1])
            outputs = model(train_seq, training=True)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=outputs))

        # gradient
        variables = model.variables
        gradients = tape.gradient(loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, clip_ratio)    # clip梯度
        optimizer.apply_gradients(zip(gradients, variables))

        if batch % 10 == 0:
            print(loss.numpy(), end=",", flush=True)
            train_losses.append(loss.numpy())

    return np.mean(train_losses)


def evaluate(model, eval_data, seq_len):
    valid_loss = []
    for batch, i in enumerate(range(0, eval_data.shape[1]-seq_len, seq_len)):
        # 获取数据
        eval_seq = tf.convert_to_tensor(eval_data[:, i:i + seq_len])
        eval_target = tf.convert_to_tensor(eval_data[:, i + 1:i + 1 + seq_len])
        assert eval_seq.shape[1] == eval_target.shape[1] == seq_len
        # eval_loss
        labels = tf.reshape(eval_target, [-1])
        outputs = model(eval_seq, training=False)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=outputs))
        valid_loss.append(loss.numpy())
    return np.mean(valid_loss)

tf.enable_eager_execution()

# 参数
batch_size = 20
embedding_dim = 500
hidden_dim = 500
num_layers = 2
sequence_length = 35
keep_prob = 0.8
clip_ratio = 0.25
Epoches = 10
logdir = "model/"

# 数据
corpus = Datasets()
train_data = corpus._divide_into_batches(corpus.train, batch_size=batch_size)
eval_data = corpus._divide_into_batches(corpus.eval, batch_size=batch_size)

print(train_data.shape)
print(eval_data.shape)

# 模型
learning_rate = tf.Variable(20.0, name="learning_rate")
model = PTBModel(corpus.vocab_size, embedding_dim, hidden_dim, 
                num_layers, sequence_length, keep_prob)

# 优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
checkpoint = tf.train.Checkpoint(
            learning_rate=learning_rate, model=model,
            optimizer=optimizer)
# checkpoint.restore(tf.train.latest_checkpoint(FLAGS.logdir))

# 训练
best_loss = None
for _ in range(Epoches):
    train_loss = train(model, optimizer, train_data, sequence_length, clip_ratio)
    eval_loss = evaluate(model, eval_data, sequence_length)
    print("训练平均损失:", train_loss, ", 测试平均损失:", eval_loss)
    # 当验证集损失下降时保存
    if not best_loss or eval_loss < best_loss:
        checkpoint.save(os.path.join(logdir, "ckpt"))
        best_loss = eval_loss
    else:   # 缩小学习率
        learning_rate.assign(learning_rate / 4.0)
        print("changing learning rate to", learning_rate.numpy())
    print("\n\n-------")
    


