import sys
import time
import json
import tqdm
import re
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
sys.path.append("..")
import utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from glob import glob
from PIL import Image

class Generator:

  def __init__(self, batch_size, buffer_size, emb_dim, epoch, units, vocab_size, features_shape, attention_features_shape,
          text_annotations_train, text_annotations_test, image_annotations_train, image_annotations_test):

    self.batch_size = batch_size
    self.buffer_size = buffer_size
    self.emb_dim = emb_dim
    self.epoch = epoch
    self.units = units
    self.vocab_size = vocab_size
    self.features_shape = features_shape
    self.attention_features_shape = attention_features_shape

    self.text_annotations_train = text_annotations_train
    self.text_annotations_test = text_annotations_test
    self.image_annotations_train = image_annotations_train
    self.image_annotations_test = image_annotations_test

  def load_image(self, image_path):
      img = tf.io.read_file(image_path)
      img = tf.image.decode_png(img, channels=3)
      img = tf.image.resize(img, (299, 299))
      img = tf.keras.applications.inception_v3.preprocess_input(img)
      return img, image_path

  def save_npy(self, image_dataset, image_features_extract_model):

      for img, path in tqdm.tqdm(image_dataset):

          batch_features = image_features_extract_model(img)
          batch_features = tf.reshape(batch_features,
                                      (batch_features.shape[0], -1, batch_features.shape[3]))

          for bf, p in tqdm.tqdm(zip(batch_features, path)):
              path_of_feature = p.numpy().decode("utf-8")
              
              np.save(path_of_feature, bf.numpy())

  def calc_max_length(self, tensor):
    return max(len(t) for t in tensor)

  def prepare_data(self):
    
      train_annotations = utils.open_json(self.text_annotations_train+"/tf_annotations_plot_train.json")
      test_annotations = utils.open_json(self.text_annotations_test+"/tf_annotations_plot_train.json")

      train_captions = []
      train_img_path = []
      test_captions = []
      test_img_path = []

      def map_func(img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8')+'.npy')
        return img_tensor, cap


      for annot in train_annotations["annotations"]:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        train_img_path.append(self.image_annotations_train + '%d.png' % (int(image_id)))
        train_captions.append(caption)

      for annot in test_annotations["annotations"]:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        test_img_path.append(self.image_annotations_test + '%d.png' % (int(image_id)))
        test_captions.append(caption)

      self.num_steps = len(train_captions) // self.batch_size


      train_captions, train_img_path = shuffle(train_captions, train_img_path, random_state=1)
      
      image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                      weights='imagenet')
      new_input = image_model.input
      hidden_layer = image_model.layers[-1].output
      self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

      encode_train = sorted(set(train_img_path))
      encode_test = sorted(set(test_img_path))

      image_train_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
      image_test_dataset = tf.data.Dataset.from_tensor_slices(encode_test)

      image_train_dataset = image_train_dataset.map(  
        self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

      image_dataset_test = image_test_dataset.map(
        self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)


      #self.save_npy(image_train_dataset, self.image_features_extract_model)
      #self.save_npy(image_test_dataset, self.image_features_extract_model)


      self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.vocab_size-1,
                                                            oov_token="<unk>",
                                                            filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
      self.tokenizer.fit_on_texts(train_captions)

      self.tokenizer.word_index['<pad>'] = 0
      self.tokenizer.index_word[0] = '<pad>'

      train_seqs = self.tokenizer.texts_to_sequences(train_captions)
      train_cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
      self.max_length = self.calc_max_length(train_seqs)

      test_seqs = self.tokenizer.texts_to_sequences(test_captions)
      test_cap_vector = tf.keras.preprocessing.sequence.pad_sequences(test_seqs, padding='post')

      dataset_train = tf.data.Dataset.from_tensor_slices((train_img_path, train_cap_vector))
      dataset_test = tf.data.Dataset.from_tensor_slices((test_img_path, test_cap_vector))

      dataset_train = dataset_train.map(lambda item1, item2: tf.numpy_function(
                map_func, [item1, item2], [tf.float32, tf.int32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

      dataset_test = dataset_test.map(lambda item1, item2: tf.numpy_function(
                map_func, [item1, item2], [tf.float32, tf.int32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

      dataset_train = dataset_train.shuffle(self.buffer_size).batch(self.batch_size)
      dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

      dataset_test = dataset_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

      return dataset_train, dataset_test, train_annotations, test_annotations, test_cap_vector

  def loss_function(self, real, pred):
      loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

      mask = tf.math.logical_not(tf.math.equal(real, 0))
      loss_ = loss_object(real, pred)

      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask

      return tf.reduce_mean(loss_)

  def train_step(self, img_tensor, target):
      
      loss = 0
      hidden = self.decoder.reset_state(batch_size=target.shape[0])
      dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)

      with tf.GradientTape() as tape:
          features = self.encoder(img_tensor)

          for i in range(1, target.shape[1]):
              predictions, hidden, _ = self.decoder(dec_input, features, hidden)
              loss += self.loss_function(target[:, i], predictions)
              dec_input = tf.expand_dims(target[:, i], 1)

      total_loss = (loss / int(target.shape[1]))
      trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
      gradients = tape.gradient(loss, trainable_variables)
      self.optimizer.apply_gradients(zip(gradients, trainable_variables))

      return loss, total_loss


  def train_model(self, dataset_train):
    self.encoder = CNN_Encoder(self.emb_dim)
    self.decoder = RNN_Decoder(self.emb_dim, self.units, self.vocab_size)

    self.optimizer = tf.keras.optimizers.Adam()

    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=self.encoder,
                              decoder=self.decoder,
                              optimizer = self.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    start_epoch = 0

    if ckpt_manager.latest_checkpoint:
      start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
      ckpt.restore(ckpt_manager.latest_checkpoint)

    for epoch in range(start_epoch, self.epoch):
      start = time.time()
      total_loss = 0

      for (batch, (img_tensor, target)) in enumerate(dataset_train):
          batch_loss, t_loss = self.train_step(img_tensor, target)
          total_loss += t_loss

          if batch % 100 == 0:
              print ('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

      if epoch % 5 == 0:
        ckpt_manager.save()

      print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                          total_loss/num_steps))
      print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


  def evaluate_single(self, image):

      hidden = self.decoder.reset_state(batch_size=1)

      temp_input = tf.expand_dims(self.load_image(image)[0], 0)
      img_tensor_test = self.image_features_extract_model(temp_input)
      img_tensor_test = tf.reshape(img_tensor_test, (img_tensor_test.shape[0], -1, img_tensor_test.shape[3]))

      features = self.encoder(img_tensor_test)

      dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
      result = []

      for i in range(self.max_length):
          predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

          predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

          result.append(self.tokenizer.index_word[predicted_id])

          if self.tokenizer.index_word[predicted_id] == '<end>':
              return result

          dec_input = tf.expand_dims([predicted_id], 0)

      return result

  def evaluate(self, annotations, test_cap_vector):

    prediction_path = os.path.join(self.text_annotations_test,"predictions")
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)


    for idx, value in tqdm.tqdm(enumerate(annotations["annotations"])):

      image_path = self.image_annotations_test + str(value["image_id"])+".png"
      real_caption = ' '.join([self.tokenizer.index_word[i] for i in test_cap_vector[idx] if i not in [0]])
      result = self.evaluate_single(image_path)

      with open(prediction_path+"/"+str(idx)+".txt","w") as f:
        json.dump({"real_caption":real_caption,
                  "prediction":' '.join(result)},f)


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):

    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):

    context_vector, attention_weights = self.attention(features, hidden)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    x = self.fc1(output)
    x = tf.reshape(x, (-1, x.shape[2]))
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))



