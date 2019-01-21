from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding


class CharRNN(Sequential):
    def __init__(self, feature_path, label_path, vocab_size, lstm_size, dropout_rate=0.5, embedding_size=128):
        super().__init__()
        self.feature_path = feature_path
        self.label_path = label_path
        self.vocab_size = vocab_size
        self.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=26))
        # because we dont know the length and we dont link to he Flatten or Dense layer
        # so we needn't know the input_length
        # self.add(Input(shape=(32, 26)))
        self.add(LSTM(lstm_size, dropout=dropout_rate, activation='sigmoid', return_sequences=True))
        self.add(LSTM(lstm_size, dropout=dropout_rate, activation='sigmoid'))
        self.add(Dense(vocab_size, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, data_stream, epochs, model_path):
        cnt = 0
        for x, y in data_stream:
            self.fit(x, y)
            cnt += 1
            if cnt == epochs:
                break
            if cnt % 10 == 0:
                self.save(model_path)
        # self.save(model_path)
