from keras.models import load_model
import numpy as np
import argparse
from utils import MakeData
from keras.utils import to_categorical


class Sample(object):
    def __init__(self, model_path, vocab_size):
        self.vocab_size = vocab_size
        self.model = load_model(model_path)

    @staticmethod
    def get_top_n(prediction, n):
        top_n = np.argsort(prediction)[-n:]
        return np.random.choice(top_n)

    def generate_sample(self, max_length, start, size):
        samples = [ch for ch in start]
        prediction = np.ones((size,))
        for ch in start:
            x = np.zeros((1, 1))
            x[0, 0] = to_categorical(ch, num_classes=self.vocab_size)
            prediction = self.model.predict(x)

        ch = self.get_top_n(prediction, 5)
        samples.append(ch)

        for steps in range(max_length):
            x = np.zeros((1, 1))
            x[0, 0] = to_categorical(ch, num_classes=self.vocab_size)
            prediction = self.model.predict(x)
            ch = self.get_top_n(prediction, 5)
            samples.append(ch)

        return np.array(samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_length', type=int, default=300,
                        help='the max length of the sample')
    parser.add_argument('--start_string', type=str, default='',
                        help='give a string to generate your sample')
    parser.add_argument('--model_path', type=str, default='./model/poetry/MyModel',
                        help='the path of your model witch you trained before')
    parser.add_argument('--vocab_path', type=str, default='./model/poetry/dictionary.txt',
                        help='the path of your dictionary')
    parser.add_argument('--word2id_path', type=str, default='./model/poetry/word2id.txt')
    args = parser.parse_args()
    vocab_size = len(open(args.vocab_path).readlines())
    MD = MakeData(word2id_path=args.word2id_path, read=True)
    start_string = MD.text2array(args.start_string)
    s = Sample(args.model_path, vocab_size)
    output = s.generate_sample(args.max_length, start_string, vocab_size)
    output = MD.array2text(output)
    print(output)
