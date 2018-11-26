import os
import sys
import numpy as np
import tensorflow as tf

# From lm_1b
import language_model.lm_1b.data_utils as data_utils

from six.moves import xrange
from google.protobuf import text_format


class GoogleLMInterface(object):
  BATCH_SIZE = 1
  NUM_TIMESTEPS = 1
  MAX_WORD_LEN = 50

  def __init__(self, path=''):
    # File Paths
    vocab_file = os.path.join(path, "language_model/data/vocab-2016-09-10.txt")
    save_dir = os.path.join(path, "language_model/output")
    pbtxt = os.path.join(path, "language_model/data/graph-2016-09-10.pbtxt")
    ckpt = os.path.join(path, "language_model/data/ckpt-*")

    # Vocabulary containing character-level information.
    self.vocab = data_utils.CharsVocabulary(vocab_file, GoogleLMInterface.MAX_WORD_LEN)

    self.targets = np.zeros([GoogleLMInterface.BATCH_SIZE, GoogleLMInterface.NUM_TIMESTEPS], np.int32)
    self.weights = np.ones([GoogleLMInterface.BATCH_SIZE, GoogleLMInterface.NUM_TIMESTEPS], np.float32)
    self.inputs = np.zeros([GoogleLMInterface.BATCH_SIZE, GoogleLMInterface.NUM_TIMESTEPS], np.int32)
    self.char_ids_inputs = np.zeros(
      [GoogleLMInterface.BATCH_SIZE, GoogleLMInterface.NUM_TIMESTEPS, self.vocab.max_word_length], np.int32)

    # Recovers the model from protobuf
    self.ckpt_file = ckpt
    self.LoadModel(pbtxt)

  def LoadModel(self, gd_file):
    """Load the model from GraphDef and Checkpoint.
    Args: gd_file: GraphDef proto text file. ckpt_file: TensorFlow Checkpoint file.
    Returns: TensorFlow session and tensors dict."""
    with tf.Graph().as_default():
      # class FastGFile: File I/O wrappers without thread locking.
      with tf.gfile.FastGFile(gd_file, 'r') as f:
        # Py 2: s = f.read().decode()
        s = f.read()
        # Serialized version of Graph
        gd = tf.GraphDef()
        # Merges an ASCII representation of a protocol message into a message.
        text_format.Merge(s, gd)

      tf.logging.info('Recovering Graph %s', gd_file)

      self.tensors = {}
      [self.tensors['states_init'], self.tensors['lstm/lstm_0/control_dependency'],
       self.tensors['lstm/lstm_1/control_dependency'], self.tensors['softmax_out'], self.tensors['class_ids_out'],
       self.tensors['class_weights_out'], self.tensors['log_perplexity_out'], self.tensors['inputs_in'],
       self.tensors['targets_in'], self.tensors['target_weights_in'], self.tensors['char_inputs_in'],
       self.tensors['all_embs'], self.tensors['softmax_weights'], self.tensors['global_step']
       ] = tf.import_graph_def(gd, {}, ['states_init',
                                        'lstm/lstm_0/control_dependency:0',
                                        'lstm/lstm_1/control_dependency:0',
                                        'softmax_out:0',
                                        'class_ids_out:0',
                                        'class_weights_out:0',
                                        'log_perplexity_out:0',
                                        'inputs_in:0',
                                        'targets_in:0',
                                        'target_weights_in:0',
                                        'char_inputs_in:0',
                                        'all_embs_out:0',
                                        'Reshape_3:0',
                                        'global_step:0'], name='')

      self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

      self.sess.run('save/restore_all', {'save/Const:0': self.ckpt_file})
      self.sess.run(self.tensors['states_init'])


  def forward(self, sess, sentence):
    """

    :param sess: tensorflow session
    :param sentence: list of words
    :return:
    """
    # Tokenize characters and words
    #sentence = sentence.encode('ascii', errors="ignore").decode()
    new_sentence = ''
    for i in np.arange(len(sentence)):
      if ord(sentence[i]) >= 256:
        new_sentence += '-'
      else:
        new_sentence += sentence[i]

    sentence = new_sentence
    print(sentence)
    word_ids = [self.vocab.word_to_id(w) for w in sentence.split()]
    char_ids = [self.vocab.word_to_char_ids(w) for w in sentence.split()]

    if sentence.find('<S>') != 0:
      sentence = '<S> ' + sentence

    embeddings = [[],[]]
    for i in xrange(len(word_ids)):
      self.inputs[0, 0] = word_ids[i]
      self.char_ids_inputs[0, 0, :] = char_ids[i]
      # Add 'lstm/lstm_0/control_dependency' if you want to dump previous layer
      # LSTM.
      lstm_emb_0, lstm_emb_1 = sess.run([self.tensors['lstm/lstm_0/control_dependency'],
                           self.tensors['lstm/lstm_1/control_dependency']],
                          feed_dict={self.tensors['char_inputs_in']: self.char_ids_inputs,
                                     self.tensors['inputs_in']: self.inputs,
                                     self.tensors['targets_in']: self.targets,
                                     self.tensors['target_weights_in']: self.weights})
      embeddings[0].append(lstm_emb_0)
      embeddings[1].append(lstm_emb_1)

    return embeddings


if __name__ == '__main__':
  google_lm = GoogleLMInterface()
  lstm_0, lstm_1 = google_lm.forward(google_lm.sess, "I am Samira نباینب are you . @ # $ % ^ & * ( ( !")
  print(np.asarray(lstm_0).shape)
