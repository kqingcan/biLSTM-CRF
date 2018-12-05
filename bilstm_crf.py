import tensorflow as tf
import numpy as np
sess = tf.Session()


def read_test_file(filename):
    test_sentences = []
    with open(filename, 'rt', encoding='utf8') as file:
        sentence = []
        tags = []
        for line in file.readlines():
            if line != "\n":
                word = line.split("\n")[0]
                sentence.append(word)
                tags.append(0)
            else:
                test_sentences.append([sentence, tags])
                sentence = []
                tags = []
    return test_sentences

class BiLSTM_CRF:

    def __init__(self, pre_file, num_units, ntags,  learning_rate):

        self.sess = sess

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_base = learning_rate
        self.learning_rate_decay = 0.99
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_base, self.global_step, 3000,
                                                        self.learning_rate_decay, staircase=True)
        self.word_map = {}
        self.dim_word = 300
        self.labels = {"B": 0, "E": 1, "I": 2, "S": 3}
        self.init_word_map(pre_file)
        self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.dim_word])
        self.tags = tf.placeholder(tf.int32, shape=[None, None])
        self.sequence_lengths = tf.placeholder(tf.int32, [None])
        # self.sequence_lengths = np.array([self.max_len for _ in range(bench_size)])
        cell_fw = tf.contrib.rnn.LSTMCell(num_units=num_units)
        cell_bw = tf.contrib.rnn.LSTMCell(num_units=num_units)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.inputs,
                                                                 sequence_length=self.sequence_lengths,
                                                                 dtype=tf.float32)
        output_fw, output_bw = outputs
        output = tf.concat([output_fw, output_bw], axis=-1)
        self.W = tf.get_variable('W', [2 * num_units, ntags])
        self.b = tf.get_variable('b', [ntags])
        m_output = tf.reshape(output, [-1, 2 * num_units])
        self.keep_prob = tf.placeholder("float")
        m_output_drop = tf.nn.dropout(m_output,self.keep_prob)
        tag_score_temp = tf.matmul(m_output_drop, self.W)+self.b
        # tag_score_temp = tf.nn.softmax(tag_score_temp)
        max_len = tf.shape(output)[1]
        self.tag_score = tf.reshape(tag_score_temp, [-1, max_len, ntags])
        # self.tag_score = tf.nn.softmax(self.tag_score)

        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.tag_score, self.tags,
                                                                                   self.sequence_lengths)
        self.decode_tags, self.best_score = tf.contrib.crf.crf_decode(self.tag_score, self.transition_params,
                                                                      self.sequence_lengths)
        regularization_cost = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4),
                                                                     tf.trainable_variables())
        self.loss = tf.reduce_mean(-log_likelihood) + regularization_cost
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
        self.optimizer = optimizer.apply_gradients(zip(grads, tvars))
        init = tf.global_variables_initializer()
        sess.run(init)
        self.saver = tf.train.Saver(tf.global_variables())


    def train(self, bench_size, benchs, max_len):
        sequence_lengths = np.array([max_len for _ in range(bench_size)])
        bench, tags = benchs
        self.sess.run(self.optimizer,
                      feed_dict={self.inputs: bench, self.tags: tags, self.sequence_lengths: sequence_lengths,self.keep_prob:0.7})
        print(self.sess.run(self.loss, feed_dict={self.inputs: bench, self.tags: tags,
                                                  self.sequence_lengths: sequence_lengths,self.keep_prob:1}))

    def test(self, batchs, sentences):
        # benchs = self.get_bench(bench_size, test_size)
        f = open('predict.utf8','w+',encoding='utf8')

        error = 0
        total = 0
        for i in range(len(batchs)):
            bench, tag = batchs[i]
            length = len(tag[0])
            sequence_lengths = np.array([length])
            sentence = sentences[i][0]
            total +=length
            # scores = self.sess.run(self.tag_score, feed_dict={self.inputs:bench,self.tags:tag})
            decode_tags = self.sess.run(self.decode_tags, feed_dict={self.inputs: bench, self.tags: tag,
                                                         self.sequence_lengths: sequence_lengths,self.keep_prob:1})
            resultSentence = sentence[0]
            for j in range(length):
                if j==0: continue
                if decode_tags[0][j] != tag[0][j]:
                    error += 1
                tag_temp = decode_tags[0][j]
                if tag_temp==0 or tag_temp==3:
                    if resultSentence[-1] !=" ":
                        resultSentence +=" "+ sentence[j]
                else:
                    resultSentence += sentence[j]
            f.write(resultSentence+"\n")
            print(resultSentence)

            # print("accuracy: ", (1 - error / total))


    def init_word_map(self, filename):
        print("init_word_map")
        with open(filename, 'r', encoding='utf8') as file:
            line_temp = file.readline()
            for line in file.readlines():
                line_array = line.split(" ")
                word = line_array[0]
                word_vec = []
                line_array.pop(0)
                for num in line_array:
                    num = float(num)
                    word_vec.append(num)
                # print(word_vec)
                self.word_map[word] = word_vec

    def get_bench(self, sentences, bench_size, train_size):
        benchs = []
        bench_num = train_size // bench_size
        for i in range(bench_num-1):
            # 0 是训练数据， 1是标准tag
            bench = []
            bench_tag = []
            for j in range(bench_size):
                sentence_index = i * bench_size + j
                sentence = sentences[sentence_index]
                tags = list(sentence[1])
                word_vecs = []
                for word in sentence[0]:
                    if word in self.word_map:
                        vec = self.word_map[word]
                        word_vecs.append(np.array(vec))
                    else:
                        vec = np.zeros(self.dim_word)
                        word_vecs.append(vec)
                bench.append(word_vecs)
                bench_tag.append(tags)
            benchs.append([bench, bench_tag])
        return benchs

    def get_one_batch(self, sentences_temp, max_sen_len, batch_size):
        batch =[]
        batch_tag = []
        for i in range(batch_size):
            sentence = sentences_temp[i]
            tags =list(sentence[1])
            word_vecs = []
            for word in sentence[0]:
                if word in self.word_map:
                    vec = self.word_map[word]
                    word_vecs.append(np.array(vec))
                else:
                    vec = np.zeros(self.dim_word)
                    word_vecs.append(vec)
            for k in range(max_sen_len - len(sentence[0])):
                word_vecs.append(np.zeros(self.dim_word))
                tags.append(0)
            batch.append(word_vecs)
            batch_tag.append(tags)
        return [batch, batch_tag]

    def read_train_file(self, train_file):
        print("read_train_file")
        sentences = []
        with open(train_file, 'rt', encoding='utf8') as file:
            sentence = []
            tags = []
            for line in file.readlines():
                if len(line.split(" ")) == 2:
                    word, tag, = line.split(" ")
                    tag = tag.strip('\n')
                    sentence.append(word)
                    tags.append(self.labels[tag])
                else:
                    sentences.append([sentence, tags])
                    sentence = []
                    tags = []
        return sentences

    def save(self, save_path):
        self.saver.save(self.sess, save_path, global_step=self.global_step)




train_num = 1
train_batch_szie = 100


biLSTM_CRF = BiLSTM_CRF("wiki_word.utf8", 100, 4, 0.01)
sentences = biLSTM_CRF.read_train_file("data/train.utf8")
# train_sentence = []
test_sentence = read_test_file("Lab3/lab_test/test.utf8")
# test_sentence = []
#
#
# for i in range(len(sentences)):
#     if i < 23000:
#         train_sentence.append(sentences[i])
#     else:
#         test_sentence.append(sentences[i])
biLSTM_CRF.saver = tf.train.import_meta_graph('model/mymodel3/mymodel3-0.meta')
biLSTM_CRF.saver.restore(sess, tf.train.latest_checkpoint("model/mymodel3/"))
# for n in range(train_num):
#     for i in range(len(train_sentence) // train_batch_szie):
#         print(i)
#         max_len = 0
#         start_index = i * train_batch_szie
#         sentences_temp = []
#         for j in range(train_batch_szie):
#             sentence = train_sentence[start_index + j]
#             max_len = max(len(sentence[0]), max_len)
#             sentences_temp.append(sentence)
#         batchs = biLSTM_CRF.get_one_batch(sentences_temp, max_len, train_batch_szie)
#         biLSTM_CRF.train(train_batch_szie, batchs, max_len)



test_batchs = biLSTM_CRF.get_bench(test_sentence, 1, len(test_sentence))

biLSTM_CRF.test(test_batchs,test_sentence)
biLSTM_CRF.save("model/mymodel3/mymodel3")