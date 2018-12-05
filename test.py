import tensorflow as tf
import numpy as np
sess = tf.Session()


class BiLSTM_CRF:

    def __init__(self, pre_file, train_file, num_units, ntags, bench_size, learning_rate):
        self.sess = sess
        self.bench_size = bench_size
        self.global_step = tf.Variable(0, trainable=False)
        self.word_map = {}
        self.max_len = 0
        self.dim_word = 300
        self.labels = {"B": 0, "E": 1, "I": 2, "S": 3}
        self.init_word_map(pre_file)
        self.sentences = self.read_train_file(train_file)
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.max_len, self.dim_word])
        self.tags = tf.placeholder(tf.int32, shape=[None, self.max_len])
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
        tag_score_temp = tf.matmul(m_output, self.W)
        # tag_score_temp = tf.nn.softmax(tag_score_temp)
        self.tag_score = tf.reshape(tag_score_temp, [-1, self.max_len, ntags])

        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.tag_score, self.tags,
                                                                                   self.sequence_lengths)
        self.decode_tags, self.best_score = tf.contrib.crf.crf_decode(self.tag_score, self.transition_params,
                                                                      self.sequence_lengths)
        self.loss = tf.reduce_mean(-log_likelihood)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        sess.run(init)

    def train(self, bench_size, benchs):
        with tf.device('/gpu:0'):
            # benchs = self.get_bench(bench_size, train_size)
            sequence_lengths = np.array([self.max_len for _ in range(bench_size)])

            i = 0
            for bench, tags in benchs:
                print(i)
                i += 1
                self.sess.run(self.optimizer,
                              feed_dict={self.inputs: bench, self.tags: tags, self.sequence_lengths: sequence_lengths})
                print(self.sess.run(self.loss, feed_dict={self.inputs: bench, self.tags: tags,
                                                          self.sequence_lengths: sequence_lengths}))
        # self.save('bilstm-crf.model')

    def test(self, benchs, sentences):
        # benchs = self.get_bench(bench_size, test_size)
        sequence_lengths = np.array([self.max_len for _ in range(1)])
        error = 0
        total = 0
        for i in range(len(benchs)):
            bench, tag = benchs[i]
            sentence = sentences[i]
            length = len(sentence[0])
            total +=length
            # scores = self.sess.run(self.tag_score, feed_dict={self.inputs:bench,self.tags:tag})
            decode_tags = self.sess.run(self.decode_tags, feed_dict={self.inputs: bench, self.tags: tag,
                                                                 self.sequence_lengths: sequence_lengths})
            for j in range(length):
                if decode_tags[0][j] != tag[0][j]:
                    error += 1
            print("accuracy: ", (1 - error / total))


    def init_word_map(self, filename):
        print("init_word_map")
        with open(filename, 'r', encoding='utf8') as file:
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

        for i in range(bench_num):
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
                for k in range(self.max_len - len(sentence[0])):
                    word_vecs.append(np.zeros(self.dim_word))
                    tags.append(0)
                bench.append(word_vecs)
                bench_tag.append(tags)
            benchs.append([bench, bench_tag])
        return benchs

    def read_train_file(self, train_file):
        print("read_train_file")
        sentences = []
        with open(train_file, 'rt', encoding='utf8') as file:
            sentence = []
            tags = []
            length = 0
            for line in file.readlines():
                if len(line.split(" ")) == 2:
                    length += 1
                    word, tag, = line.split(" ")
                    tag = tag.strip('\n')
                    sentence.append(word)
                    tags.append(self.labels[tag])
                else:
                    if length > self.max_len:
                        self.max_len = length
                    length = 0
                    sentences.append([sentence, tags])
                    sentence = []
                    tags = []
        return sentences

    def save(self, save_path):
        self.saver.save(self.sess, save_path, global_step=self.global_step)


biLSTM_CRF = BiLSTM_CRF("train.txt", "data/train.utf8", 50, 4, 50, 0.08)
train_sentence = []
test_sentence = []
for i in range(len(biLSTM_CRF.sentences)):
    if i < 20000:
        train_sentence.append(biLSTM_CRF.sentences[i])
    else:
        test_sentence.append(biLSTM_CRF.sentences[i])
train_benchs = biLSTM_CRF.get_bench(train_sentence, 50, 20000)
test_benchs = biLSTM_CRF.get_bench(test_sentence, 1, len(test_sentence))

biLSTM_CRF.train(50,train_benchs)
biLSTM_CRF.test(test_benchs,test_sentence)
