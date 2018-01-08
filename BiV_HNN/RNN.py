import tensorflow as tf
import numpy as np
import pdb

class RNN():

  def __init__(self, cell_size, input_size, embedding, keep_prob=1.0, bool_has_attention=False,
               bool_bidirection=True, random_seed=1):
    self._cell_size = cell_size
    self._input_size = input_size
    self._has_attention = bool_has_attention
    self._bidirection = bool_bidirection
    self._embedding = embedding
    self._random_seed = random_seed
    self._keep_prob = keep_prob

    assert not self._has_attention

    # create the cell
    with tf.variable_scope("GRU_cell", initializer=tf.random_uniform_initializer(
            (-1) * np.sqrt(6.0 / (self._cell_size * 2)),
        np.sqrt(6.0 / (self._cell_size * 2)), dtype=tf.float32)):
      self._cell = tf.contrib.rnn.GRUCell(self._cell_size)

  def RNN_run(self, input_sentence_w_embedding, input_actual_length):
    """

    Args:
      input_sentence_w_embedding: a list of [bs, self._input_size] 2D Tensors. The embedded input sentences.
      input_actual_length: a bs-sized list of integers, the actual length of each sentence.
      bool_bidirection: set to True for bidirectional RNN. Default True.

    Returns:
      outputs: a list of [bs, self._cell_size] (or [bs, self._cell_size * 2]) 2D Tensors, the outputs of
        forward (or bidirectional) RNN.

    """
    with tf.variable_scope("GRU_RNN"):
      # dropout
      if self._keep_prob < 1:
        input_sentence_w_embedding = [tf.nn.dropout(item, self._keep_prob, seed=self._random_seed)
                                      for item in input_sentence_w_embedding]

      input_sentence_w_embedding = tf.stack(input_sentence_w_embedding, axis=0)  # [time_step, bs, self._input_size]

      if self._bidirection:
        outputs, states = tf.nn.bidirectional_dynamic_rnn(self._cell, self._cell, input_sentence_w_embedding,
                                                                sequence_length=input_actual_length, time_major=True,
                                                                dtype=tf.float32)
        outputs = tf.concat(outputs, axis=2)

      else:
        outputs, states = tf.nn.dynamic_rnn(self._cell, input_sentence_w_embedding,
                                               sequence_length=input_actual_length, time_major=True,
                                                dtype=tf.float32)

    outputs = tf.unstack(outputs, axis=0)
    return outputs, states


  def RNN_semantic(self, input_sentence, input_actual_length):
    """
    Args:
      input_sentence: a list of bs-length lists of integers, the index of tokens.
      input_actual_length: a bs-sized list of integers, the actual length of each sentence.

    Returns:
      sem_outputs: a 2D tensor of shape [batch_size, self._cell_size or self._cell_size*2].
    """
    with tf.variable_scope("GRU_RNN"):

      batch_size = input_sentence[0].get_shape()[0].value

      inputs = [] # inputs is a list of [bs, num_units] Tensors
      for j in range(len(input_sentence)):
        item_j = tf.nn.embedding_lookup(self._embedding, input_sentence[j])
        # dropout
        if self._keep_prob < 1:
          item_j = tf.nn.dropout(item_j, keep_prob=self._keep_prob, seed=self._random_seed)
        inputs.append(item_j)
      inputs = tf.stack(inputs, axis=0)

      if self._bidirection:
        outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
          self._cell, self._cell, inputs, sequence_length=input_actual_length, time_major=True, dtype=tf.float32)

        last_forward_states, last_backward_states = last_states
        sem_outputs = tf.concat([last_forward_states, last_backward_states], axis=1)
        sem_outputs = tf.reshape(sem_outputs, [batch_size, self._cell_size*2])

        return sem_outputs

      else:
        outputs, states = tf.nn.dynamic_rnn(self._cell, inputs, sequence_length=input_actual_length,
                                                     time_major=True, dtype=tf.float32)

        states = tf.reshape(states, [batch_size, self._cell_size])
        return states


