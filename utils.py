import numpy as np
from keras.utils import to_categorical
import pickle


def data_stream(feature_path, label_path, vocab_size):
    features = np.load(feature_path)
    labels = np.load(label_path)
    while True:
        # TODO: shuffle
        for feature, label in zip(features, labels):
            label = to_categorical(label, num_classes=vocab_size)
            yield feature, label


class MakeData(object):
    def __init__(self, raw_path='', dict_path='', word2id_path='',
                 feature_path='', label_path='', low_frequency=0, read=False):
        self.read = read
        if read:
            self.word2id = {}
            self.id2word = {}
            with open(word2id_path, 'rb') as f:
                w2i = pickle.load(f)
            for i in range(len(w2i)):
                self.word2id[w2i[i]] = i
                self.id2word[i] = w2i[i]

        else:
            self.path = raw_path
            self.save_path = dict_path
            self.word2id_path = word2id_path
            self.low_frequency = low_frequency
            self.feature_path = feature_path
            self.label_path = label_path
            self.words = {}
            self.word2id = {}
            self.id2word = {}

    def pretreatment_data(self):
        file = open(self.path, 'r', encoding='utf-8')
        content = file.readlines()
        file.close()
        for line in content:
            for i in range(len(line)):
                if line[i] not in self.words.keys():
                    self.words[line[i]] = 1
                else:
                    self.words[line[i]] += 1

    def delete_low_frequency(self):
        if self.low_frequency != -1:
            self.words = {word: frequency for word, frequency in self.words.items() if frequency > self.low_frequency}
        index = 0
        for word in self.words:
            self.word2id[word] = index
            index += 1
        w2i = [word for word in self.words]
        with open(self.word2id_path, 'wb') as f:
            pickle.dump(list(w2i), f)

    def save_dictionary(self):
        dictionary = open(self.save_path, 'w', encoding='utf-8')
        for word, frequency in self.words.items():
            dictionary.write(word + " " + str(frequency) + "\n")
        dictionary.close()

    @property
    def vocab_size(self):
        if self.read:
            return len(self.word2id)
        return len(self.words.keys())

    def get_id(self, word):
        return self.word2id[word]

    def get_word(self, index):
        return self.id2word[index]

    def text2array(self, text):
        array = []
        for i in range(len(text)):
            array.append(self.get_id(text[i]))
        return array

    def array2text(self, array):
        text = ""
        for item in array:
            text += self.get_word(item)
        return text

    def make_data(self, text_array, batch_size, steps):
        text_array = np.array(text_array)
        unit = batch_size * steps
        batches = int(len(text_array) / unit)
        text_array = text_array[:unit * batches]
        text_array = text_array.reshape((batch_size, -1))
        print(text_array.shape)
        feature = []
        label = []
        for n in range(0, text_array.shape[1], steps):
            # print("(n: n + steps) " + str(n) + ": " + str(n + steps))
            x = text_array[:, n: n + steps]
            if (n + steps) != text_array.shape[1]:
                y = text_array[:, n + steps]
            else:
                y = text_array[:, 0]
            feature.append(x)
            label.append(y)
        feature = np.array(feature)
        feature = np.squeeze(feature)
        label = np.array(label)
        label = np.squeeze(label)
        print("feature\'s shape: " + str(feature.shape))
        print("label\'s shape: " + str(label.shape))
        np.save(self.feature_path, feature)
        np.save(self.label_path, label)


if __name__ == '__main__':
    MD = MakeData('./data/poetry.txt',
                  './data/vectors.npy',
                  './data/word2id_path',
                  './data/feature_path',
                  './data/label_path',
                  1)
