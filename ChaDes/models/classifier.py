import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model, save_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.applications.resnet import preprocess_input, ResNet50
from sklearn.metrics import confusion_matrix, classification_report


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

class ChartClassifier():
    def __init__(self, config, prefix):
        self.config = config
        self.prefix = prefix
        self.model_path = os.path.join(self.config.default_path+"/"+self.config.model, self.prefix)

    def read_data(self):

        if self.config.train_frac < 1.0:
            valid_frac = 1.0 - self.config.train_frac
        else:
            valid_frac = False

        train_gen = ImageDataGenerator(rescale=1./255,
                                       validation_split=valid_frac)

        train_generator = train_gen.flow_from_directory(
            self.config.X_train_path,
            target_size=self.config.target_size,
            batch_size=self.config.batch_size,
            color_mode=self.config.color_mode,
            class_mode='categorical',
            subset="training"
        )

        valid_gen = ImageDataGenerator(rescale=1./255)

        validation_generator = valid_gen.flow_from_directory(
            self.config.X_val_path,
            target_size=self.config.target_size,
            batch_size=self.config.batch_size,
            color_mode=self.config.color_mode,
            class_mode='categorical'
        )
        return train_generator, validation_generator

    def _build_model(self):
        print("Selected mode: CNN")
        filters = 32
        model = Sequential()
        if self.config.color_mode == "rgb":
            model.add(Conv2D(filters, (3, 3), input_shape=(164, 164, 3)))
        else:
            model.add(Conv2D(filters, (3, 3), input_shape=(164, 164, 1)))

        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        for layer in range(self.config.n_hidden_layers-1):
            model.add(Conv2D(filters, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            filters *= 2

        model.add(Flatten())
        model.add(Dense(filters))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.config.num_classes, activation="softmax"))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        return model

    def _build_svm(self):

        print("Selected mode: SVM")
        filters = 32
        model = Sequential()
        if self.config.color_mode == "rgb":
            model.add(Conv2D(filters, (3, 3), input_shape=(164, 164, 3)))
        else:
            model.add(Conv2D(filters, (3, 3), input_shape=(164, 164, 1)))

        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        for layer in range(self.config.n_hidden_layers-1):
            model.add(Conv2D(filters, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            filters *= 2

        model.add(Flatten())
        model.add(Dense(filters))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.config.num_classes, W_regularizer=l2(0.01)))
        model.add(Activation('linear'))
        model.compile(loss='hinge',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        return model

    def load_model(self):
        self.model = load_model(os.path.join(self.model_path, "model.h5"))

    def train(self):    
        train_generator, validation_generator = self.read_data()
        if self.config.model == "cnn":
            self.model = self._build_model()
        elif self.config.model == "svm":
            self.model = self._build_svm()

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)

        history = self.model.fit_generator(train_generator,
                                 steps_per_epoch=self.config.steps_per_epoch,
                                 epochs=self.config.epoch,
                                 validation_data=validation_generator,
                                 validation_steps=self.config.validation_steps,
                                 callbacks=[es],
                                 workers=0)

        self.plot_learning_curves(history)

    def save_model(self):
        save_model(self.model, os.path.join(self.model_path, "model.h5")) 

    def predict(self, X_test_path):

        test_gen = ImageDataGenerator(rescale=1./255)

        test_generator = test_gen.flow_from_directory(
            X_test_path,
            target_size=self.config.target_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            color_mode=self.config.color_mode,
            shuffle=False
        )

        corpora = X_test_path.split("data/")[1][:4]

        probs = self.model.predict_generator(test_generator)
        prediction = np.argmax(probs, axis=1)

        #probabilites
        probs = pd.DataFrame(probs, columns=test_generator.class_indices.keys())
        probs.to_csv(os.path.join(self.model_path, corpora + "_probabilities.csv"))

        #confusion_matrix
        conf_matrix = pd.DataFrame(
            confusion_matrix(test_generator.classes, prediction),
            index=test_generator.class_indices.keys(),
            columns=test_generator.class_indices.keys()
        )
        conf_matrix.to_csv(os.path.join(self.model_path, corpora + "_confusion_matrix.csv"))

        #classification_report
        report = classification_report(test_generator.classes, prediction, output_dict=True, target_names=test_generator.class_indices.keys())
        pd.DataFrame(report).transpose().to_csv(os.path.join(self.model_path, corpora + "_classification_report.csv"))


    def plot_learning_curves(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(self.model_path, "_learning_curves_acc.png"))
        plt.close()
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.savefig(os.path.join(self.model_path, "_learning_curves_loss.png"))


class ResNetClassifier():
    def __init__(self, config, prefix):
        self.config = config
        self.prefix = prefix
        self.model_path = os.path.join(self.config.default_path+"/resnet", self.prefix)

    def read_data(self):

        if self.config.train_frac < 1.0:
            valid_frac = 1.0 - self.config.train_frac
        else:
            valid_frac = False

        train_gen = ImageDataGenerator(rescale=1./255,
                                       validation_split=valid_frac)

        train_generator = train_gen.flow_from_directory(
            self.config.X_train_path,
            target_size=self.config.target_size,
            batch_size=self.config.batch_size,
            color_mode=self.config.color_mode,
            class_mode='categorical',
            subset="training"
        )

        valid_gen = ImageDataGenerator(rescale=1./255)

        validation_generator = valid_gen.flow_from_directory(
            self.config.X_val_path,
            target_size=self.config.target_size,
            batch_size=self.config.batch_size,
            color_mode=self.config.color_mode,
            class_mode='categorical'
        )
        return train_generator, validation_generator

    def _build_model(self):

        model = Sequential()
        model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(2048, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(int(self.config.num_classes), activation='softmax'))

        model.layers[0].trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def load_model(self):
        self.model = load_model(os.path.join(self.model_path, "model.h5"))

    def train(self):    
        train_generator, validation_generator = self.read_data()
        self.model = self._build_model()

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)

        history = self.model.fit_generator(
            train_generator,
            steps_per_epoch=self.config.steps_per_epoch,
            epochs=self.config.epoch,
            validation_data=validation_generator,
            validation_steps=self.config.validation_steps,
            callbacks=[es],
            workers=0
        )

        self.plot_learning_curves(history)

    def save_model(self):
        save_model(self.model, os.path.join(self.model_path, "model.h5")) 

    def predict(self, X_test_path):

        test_gen = ImageDataGenerator(rescale=1./255)

        test_generator = test_gen.flow_from_directory(
            X_test_path,
            target_size=self.config.target_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            color_mode=self.config.color_mode,
            shuffle=False
        )

        corpora = X_test_path.split("data/")[1][:4]

        probs = self.model.predict_generator(test_generator)
        prediction = np.argmax(probs, axis=1)

        #probabilites
        probs = pd.DataFrame(probs, columns=test_generator.class_indices.keys())
        probs.to_csv(os.path.join(self.model_path, corpora + "_probabilities.csv"))

        #confusion_matrix
        conf_matrix = pd.DataFrame(
            confusion_matrix(test_generator.classes, prediction),
            index=test_generator.class_indices.keys(),
            columns=test_generator.class_indices.keys()
        )
        conf_matrix.to_csv(os.path.join(self.model_path, corpora + "_confusion_matrix.csv"))

        #classification_report
        report = classification_report(test_generator.classes, prediction, output_dict=True, target_names=test_generator.class_indices.keys())
        pd.DataFrame(report).transpose().to_csv(os.path.join(self.model_path, corpora + "_classification_report.csv"))


    def plot_learning_curves(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(self.model_path, "_learning_curves_acc.png"))
        plt.close()
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.savefig(os.path.join(self.model_path, "_learning_curves_loss.png"))
