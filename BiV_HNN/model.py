# model.py 

""" The Hierarchical Neural Network Model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random, sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pickle

import HNN
from RNN import RNN

import pdb


class HNNModel(object):
  """ Hierarchical Neural Network Model. """
  def __init__(self, buckets, config, forward_only=False):
    self._buckets = buckets
    self._buckets_text_max = (max([i for i,_,_,_ in buckets]), max([j for _,j,_,_ in buckets]))
    self._buckets_code_max = (max([i for _,_,i,_ in buckets]), max([j for _,_,_,j in buckets]))
    self._batch_size = config.batch_size
    self._keep_prob = config.keep_prob
    self._learning_rate = tf.Variable(float(config.learning_rate), trainable=False)
    self._global_step = tf.Variable(0, trainable=False)


    # module setting
    self._HNN_variant = config.HNN_variant
    self._text_model_setting = config.text_model
    self._code_model_setting = config.code_model
    self._query_model_setting = config.query_model
    # block level setting
    self._block_setting = []
    if self._text_model_setting.type != 0:
      if self._HNN_variant == 1:
        self._block_size = config.block_size
        self._input_size = config.input_size
        self._block_setting = [self._block_size, self._input_size]
      elif self._HNN_variant == 2:
        self._num_mlp_layers = config.num_mlp_layers
        self._mlp_layer_units = config.mlp_layer_units
        self._block_setting = [self._num_mlp_layers, self._mlp_layer_units]
    # code
    if self._code_model_setting.type != 0:
      self._code_num_mlp_layers = config.code_num_mlp_layers
      self._code_mlp_layer_units = config.code_mlp_layer_units
    else:
      self._code_num_mlp_layers = 0
      self._code_mlp_layer_units = []

    # module generation and placeholder definition
    with tf.variable_scope("text_model"):
      if self._text_model_setting.type == 1:
        self.text_blocks = [] # a list of lists of [bs or (bs,text_model_setting._input_size)]
        self.text_block_actual_lengths = []

        # create word embedding
        self.text_word_embedding = tf.get_variable("word_embedding", shape=[
            self._text_model_setting.word_vocab_size, self._text_model_setting.input_size])
        text_model = RNN(self._text_model_setting.cell_size,
                       self._text_model_setting.input_size,
                       self.text_word_embedding,
                       keep_prob=self._keep_prob,
                       bool_has_attention=self._text_model_setting.bool_has_attention,
                       bool_bidirection=self._text_model_setting.bool_bidirection,
                       random_seed = config.random_seed)
        for i in xrange(self._buckets_text_max[0]):
          word_inputs = []  # a buckets_text[-1][1] length of list of batch_size arrays
          for j in xrange(self._buckets_text_max[1]):  # bucket size of the word-level encoder
            word_inputs.append(tf.placeholder(tf.int32, shape=[self._batch_size],
                                          name="sentence{0}_word{1}".format(i, j)))
          self.text_blocks.append(word_inputs)
          self.text_block_actual_lengths.append(tf.placeholder(tf.int32, shape=[self._batch_size],
                                                         name="sentence{0}_text_length".format(i)))
      else:
        text_model = None
        self.text_blocks = None
        self.text_block_actual_lengths = None

    # code model
    with tf.variable_scope("code_model"):
      if self._code_model_setting.type == 1:
        self.code_blocks = [] # a list of lists of [bs or (bs, c_model.input_size)]
        self.code_block_actual_lengths = []
        self.code_token_embedding = tf.get_variable("token_embedding", shape=[
          self._code_model_setting.word_vocab_size, self._code_model_setting.input_size
        ])
        code_model = RNN(self._code_model_setting.cell_size,
                         self._code_model_setting.input_size,
                         self.code_token_embedding,
                         keep_prob=self._keep_prob,
                         bool_has_attention=self._code_model_setting.bool_has_attention,
                         bool_bidirection=self._code_model_setting.bool_bidirection,
                         random_seed=config.random_seed)
        for i in xrange(self._buckets_text_max[0]-1):
          code_inputs = []
          for j in xrange(self._buckets_code_max[1]):
            code_inputs.append(tf.placeholder(tf.int32, shape=[self._batch_size],
                                              name="sentence{0}_code{1}".format(i, j)))
          self.code_blocks.append(code_inputs)
          self.code_block_actual_lengths.append(tf.placeholder(tf.int32, shape=[self._batch_size],
                                                           name="sentence{0}_code_length".format(i)))
      else:
        code_model = None
        self.code_blocks = None
        self.code_block_actual_lengths = None

    # query model
    if self._query_model_setting.type == 1:
      self.queries = []
      for i in xrange(self._buckets_code_max[0]):
        self.queries.append(tf.placeholder(tf.int32, shape=[self._batch_size],
                                           name="query_word{0}".format(i)))
      # actual length of the query
      self.query_actual_lengths = tf.placeholder(tf.int32, shape=[self._batch_size],
                                                 name="query_lengths")

      if self._query_model_setting == self._text_model_setting:
        query_model = text_model
      else:
        with tf.variable_scope("query_model"):
          self.query_word_embedding = tf.get_variable("word_embedding", shape=[
            self._query_model_setting.word_vocab_size, self._query_model_setting.input_size
          ])
          query_model = RNN(self._query_model_setting.cell_size,
                            self._query_model_setting.input_size,
                            self.query_word_embedding,
                            keep_prob=self._keep_prob,
                            bool_has_attention=self._query_model_setting.bool_has_attention,
                            bool_bidirection=self._query_model_setting.bool_bidirection,
                            random_seed=config.random_seed)

    else:
      query_model = None
      self.queries = None
      self.query_actual_lengths = None

    self.code_label_targets = tf.placeholder(tf.int32, shape=[self._batch_size],
                                             name="code_label_targets")
    self.code_selected_indices = tf.placeholder(tf.int32, shape=[self._batch_size],
                                                name="code_selected_indices")

    # placeholder for actual block length
    self.block_sequence_actual_lengths = tf.placeholder(tf.int32, shape=[self._batch_size],
                                                        name="block_sequence_actual_lengths")

    # feed to the model
    self.losses, self.predictions, self.token_level_outputs, self.summaries =\
      HNN.model_with_buckets(buckets, self._HNN_variant, self._block_setting, self._code_num_mlp_layers,
                             self._code_mlp_layer_units, text_model, self._text_model_setting, code_model,
                             self._code_model_setting, query_model, self._query_model_setting, self.text_blocks,
                             self.text_block_actual_lengths, self.code_blocks, self.code_block_actual_lengths,
                             self.queries, self.query_actual_lengths, self.code_label_targets,
                             self.code_selected_indices, self.block_sequence_actual_lengths,
                             config.keep_prob, random_seed=config.random_seed)

    self.params = tf.trainable_variables()
    for params_i in self.params:
      print(params_i)
    print("\n"); sys.stdout.flush()

    if not forward_only:
      # update params
      opt = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
      self._train_op = []
      for b in xrange(len(buckets)):
        self._train_op.append(opt.minimize(self.losses[b], global_step=self._global_step))

    self.saver = tf.train.Saver(tf.global_variables())


  def step(self, session, text_blocks, text_block_actual_lengths, code_blocks, code_block_actual_length,
           queries, query_actual_lengths, code_label_targets, code_selected_indices,
           block_sequence_actual_lengths, bucket_id, forward_only=False):
    """ run a step of HNN model.

    Args:
      session: the running session.
      bucket_id: the selected bucket id.
      forward_only: set to True for testing, otherwise set it to False.

    Returns:
    """

    text_block_length, text_word_length, query_word_length, code_token_length = self._buckets[bucket_id]

    input_feed = {}
    # text blocks
    if self._text_model_setting.type != 0:
      for i in xrange(text_block_length):
        for j in xrange(text_word_length):
          input_feed[self.text_blocks[i][j].name] = text_blocks[i][j]
        input_feed[self.text_block_actual_lengths[i].name] = text_block_actual_lengths[i]

    # code blocks
    if self._code_model_setting.type != 0:
      for i in xrange(text_block_length - 1):
        for j in xrange(code_token_length):
          input_feed[self.code_blocks[i][j].name] = code_blocks[i][j]
        input_feed[self.code_block_actual_lengths[i].name] = code_block_actual_length[i]

    if self._query_model_setting.type != 0:
      for i in xrange(query_word_length):
        input_feed[self.queries[i].name] = queries[i]
      input_feed[self.query_actual_lengths.name] = query_actual_lengths

    input_feed[self.code_label_targets.name] = code_label_targets
    input_feed[self.code_selected_indices.name] = code_selected_indices
    input_feed[self.block_sequence_actual_lengths.name] = block_sequence_actual_lengths

    # outputs
    if not forward_only: # train
      output_feed = [self.losses[bucket_id],
                     self._train_op[bucket_id],
                     self.predictions[bucket_id]]
      # output_feed.append(tf.summary.merge(self.summaries[bucket_id]))

    else: # validation or test
      output_feed = [self.losses[bucket_id],
                     self.predictions[bucket_id]]

    outputs = session.run(output_feed, input_feed)

    if not forward_only:
      return (outputs[0], # loss
              outputs[2] # prediction, a 2d tensor of shape [bs, 2]
              # outputs[3] # merged summary
              )
    else:
      return (outputs[0], # loss
              outputs[1] # prediction
              )


  def get_batch(self, data, bucket_id, text_embedding=None, code_embedding=None, query_embedding=None,
                bool_shuffle=True):
    """ Get a batch of data. """

    text_block_length, text_word_length, query_word_length, code_token_length = self._buckets[bucket_id]

    if bool_shuffle:
      samples = random.sample(data[bucket_id], self._batch_size)  # make sure not overlapping.
    else:
      assert len(data[bucket_id]) >= self._batch_size
      samples = data[bucket_id][:self._batch_size]

    def process_instance(instance, embedding, target, is_vec):
      if is_vec:
        instance2id = [embedding[i] for i in instance]
        target.append(np.array(instance2id))
      else:
        target.append(np.array(instance))

    # inputs of shape [bs, text_block_length, text_word_length]
    def process_matrix(inputs, trans1_length, trans2_length, embedding, is_vec=False):
      inputs_trans1 = np.split(inputs, trans1_length, axis=1) # a text_block_length list of [bs, text_word_length]
      processed_inputs = []
      for item in inputs_trans1:
        item_trans2 = np.split(np.squeeze(item, axis=1), trans2_length, axis=1) # a text_word_length list of [bs]
        processed_item = [] # a text_word_length list of [bs or (bs, embedding_size)]
        for iitem in item_trans2:
          process_instance(np.squeeze(iitem, axis=1), embedding, processed_item, is_vec=is_vec)
        processed_inputs.append(processed_item)

      return processed_inputs


    # text block
    if self._text_model_setting.type == 0:
      text_blocks = None
      text_block_actual_lengths = None
    else:
      text_blocks = process_matrix(np.array([samples_term[1] for samples_term in samples]),
                                   text_block_length, text_word_length, text_embedding,
                                   is_vec=False)
      assert(np.array(text_blocks).shape[:3] == (text_block_length, text_word_length, self._batch_size))
      if self._text_model_setting.type == 1:
        text_block_actual_lengths = np.array([np.squeeze(item, axis=1) for item in
                                              np.split(np.array([samples_term[2] for samples_term in samples]),
                                                       text_block_length, axis=1)])
      else:
        text_block_actual_lengths = np.array([np.squeeze(item, axis=1) for item in
                                              np.split(np.array([[i-2 if i>0 else 0 for i in samples_term[2]]
                                                                 for samples_term in samples]),
                                                       text_block_length, axis=1)])

    # code block
    if self._code_model_setting.type == 0:
      code_blocks = None
      code_block_actual_lengths = None
    else:
      code_blocks = process_matrix(np.array([samples_term[3] for samples_term in samples]),
                                   text_block_length-1, code_token_length, code_embedding,
                                   is_vec=False)
      assert(np.array(code_blocks).shape[:3] == (text_block_length-1, code_token_length, self._batch_size))
      if self._code_model_setting.type == 1:
        code_block_actual_lengths = np.array([np.squeeze(item, axis=1) for item in
                                     np.split(np.array([samples_term[4] for samples_term in samples]),
                                              text_block_length - 1, axis=1)])
      else:
        code_block_actual_lengths = np.array([np.squeeze(item, axis=1) for item in
                                              np.split(np.array([[i-2 if i>0 else 0 for i in samples_term[4]]
                                                                 for samples_term in samples]),
                                                       text_block_length - 1, axis=1)])
      assert(np.array(code_block_actual_lengths).shape == (text_block_length-1, self._batch_size))

    # queries
    if self._query_model_setting.type == 0:
      queries = None
      query_actual_lengths = None
    else:
      queries = []
      pre_queries = np.split(np.array([samples_term[5] for samples_term in samples]), query_word_length, axis=1)
      for item in pre_queries:
        item = np.squeeze(item, axis=1) # [bs]
        process_instance(item, query_embedding, queries, is_vec=False)

      if self._query_model_setting.type == 1:
        query_actual_lengths = np.array([samples_term[6] for samples_term in samples]) # [bs]
      else:
        query_actual_lengths = np.array([samples_term[6]-2 if samples_term[6]>0 else 0
                                         for samples_term in samples])  # [bs]
      assert(np.array(queries).shape[:2] == (query_word_length, self._batch_size))

    # code label
    code_label_targets = np.array([samples_term[7] for samples_term in samples]) # [bs]
    code_selected_indices = np.array([samples_term[8] for samples_term in samples])  # [bs]

    block_sequence_actual_length = np.array([samples_term[9] for samples_term in samples])  # [bs]


    return text_blocks, text_block_actual_lengths, code_blocks, code_block_actual_lengths,\
      queries, query_actual_lengths, code_label_targets, code_selected_indices, block_sequence_actual_length

