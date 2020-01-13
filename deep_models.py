from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten


def NN_model(X_train, y_train, X_test, y_test, ndim, embedding_dim):
    model = Sequential()
    model.add(Embedding(input_dim = ndim, 
                           output_dim = embedding_dim, 
                           input_length = len(X_train)))
    model.add(Flatten())
    model.add(Dense(1024, input_dim = ndim, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs = 150, batch_size = 1000)

    _, train_accuracy = model.evaluate(X_train, y_train)

    predictions = model.predict(X_test)
    _, test_accuracy = model.evaluate(X_test, y_test)

    return predictions, train_accuracy, test_accuracy