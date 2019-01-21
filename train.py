import argparse
import os
from utils import MakeData, data_stream
from model import CharRNN
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
import tensorflow as tf

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='data/poetry.txt',
                    help='data directory containing input.txt with training example')
parser.add_argument('--model_path', type=str, default='model/',
                    help='directory to store the checkpoint models')
parser.add_argument('--model_name', type=str, default='poetry/')
parser.add_argument('--lstm_size', type=int, default=128,
                    help='size of hidden state of lstm')
parser.add_argument('--steps', type=int, default=26,
                    help='the length of each sentence')
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--layers_nums', type=int, default=2)
parser.add_argument('--embedding_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()


def run():
    model_path = os.path.join(args.model_path, args.model_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    dict_path = os.path.join(model_path, 'dictionary.txt')
    word2id_path = os.path.join(model_path, 'word2id.txt')
    feature_path = os.path.join(model_path, 'feature.npy')
    label_path = os.path.join(model_path, 'label.npy')

    print("data_path " + args.data_path)
    print("model_path " + model_path)
    print("dict_path " + dict_path)
    print("word2id_path " + word2id_path)
    print("feature_path " + feature_path)
    print("label_path " + label_path)

    data_maker = MakeData(raw_path=args.data_path,
                          dict_path=dict_path,
                          word2id_path=word2id_path,
                          feature_path=feature_path,
                          label_path=label_path,
                          low_frequency=1)
    data_maker.pretreatment_data()
    data_maker.delete_low_frequency()
    file = open(args.data_path, 'r', encoding='utf-8')
    content = file.read()
    file.close()
    text_array = data_maker.text2array(content)
    data_maker.save_dictionary()
    data_maker.make_data(text_array, args.batch_size, args.steps)

    model = CharRNN(vocab_size=data_maker.vocab_size,
                    feature_path=feature_path,
                    label_path=label_path,
                    lstm_size=args.lstm_size,
                    dropout_rate=args.dropout_rate,
                    embedding_size=args.embedding_size)

    model.train(data_stream=data_stream(feature_path, label_path, data_maker.vocab_size),
                epochs=args.epochs, model_path=os.path.join(model_path, 'MyModel'))


if __name__ == '__main__':
    run()
