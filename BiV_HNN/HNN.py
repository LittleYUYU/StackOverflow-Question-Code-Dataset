# HNN.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def symmetric_variable_summaries(var, metrics=None):
  """
  Set summaries for a symmetric variable, e.g., the output of tanh.
  Args:
    var: tensorflow variable.
    metrics: set a list of metrics for evaluation. Support: mean, stddev, max, min, histogram.
      Default: mean, stddev.

  Returns:
    summaries: a list of summaries.

  """
  summaries = []

  if metrics is None:
    metrics = ["mean-stddev"]

  with tf.name_scope("symmatric_value_summary"):
    with tf.name_scope("positive_value_summary"):
      # first, get positive or zero values
      where_pos = tf.where(tf.greater_equal(var, tf.constant(0, dtype=tf.float32)))
      pos_elements = tf.gather_nd(var, where_pos)
      summaries.extend(variable_summaries(pos_elements, metrics=metrics))

    with tf.name_scope("negative_value_summary"):
      # then, get negative values
      where_neg = tf.where(tf.less(var, tf.constant(0, dtype=tf.float32)))
      neg_elements = tf.gather_nd(var, where_neg)
      summaries.extend(variable_summaries(neg_elements, metrics=metrics))

  return summaries


def variable_summaries(var, metrics=None):
  """
  Set summaries for a variable: mean, std deviation, max, min, histogram.

  Args:
    var: a Tensorflow variable.
    metrics: set a list of metrics for evaluation. Support: mean, stddev, max, min, histogram.
      Default: mean, stddev.

  Returns:
    summaries: a list of summaries.

  """
  summaries = []

  if metrics is None:
    metrics = ["mean-stddev"]

  with tf.name_scope("summary"):

    if "mean-stddev" in metrics or "mean" in metrics:
      mean = tf.reduce_mean(var)
      summaries.append(tf.summary.scalar('mean', mean))
      if "mean-stddev" in metrics:
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        summaries.append(tf.summary.scalar('stddev', stddev))

    if "min" in metrics:
      summaries.append(tf.summary.scalar('min', tf.reduce_min(var)))

    if "max" in metrics:
      summaries.append(tf.summary.scalar('max', tf.reduce_max(var)))

    if "histogram" in metrics:
      summaries.append(tf.summary.histogram('histogram', var))

  return summaries


def MLP(num_layers, layer_units, input, input_dim, scope, keep_prob, random_seed=1):
  """
  A MLP module.
  Args:
    num_layers: number of layers.
    layer_units: a list of integers, the unit size in each layer.
    input: a 2D Tensor of shape [batch_size, input_dim].

  Returns:
    output: a 2D Tensor of shape [batch_size, last_int_in_layer_units].
  """
  with tf.variable_scope(scope):
    summaries = []

    layer_input_dim = input_dim
    layer_input = input
    for layer_idx in range(num_layers):
      # dropout
      if keep_prob < 1:
        layer_input = tf.nn.dropout(layer_input, keep_prob=keep_prob, seed=random_seed)

      layer_output_dim = layer_units[layer_idx]
      with tf.variable_scope("Layer_%d" % layer_idx):
        weight = tf.get_variable("weight", shape=[layer_input_dim, layer_output_dim],
                                 initializer=tf.random_uniform_initializer(
                                   (-1) * np.sqrt(6.0 / (layer_input_dim + layer_output_dim)),
                                   np.sqrt(6.0 / (layer_input_dim + layer_output_dim))
                                 ))
        bias = tf.get_variable("bias", shape=[layer_output_dim],
                               initializer=tf.zeros_initializer())
        layer_input = tf.nn.tanh(tf.matmul(layer_input, weight) + bias)
        layer_input_dim = layer_output_dim

        # record
        summaries.extend(symmetric_variable_summaries(layer_input))

    return layer_input, summaries


def hierarchical_neural_network(HNN_variant, block_setting, code_num_mlp_layers, code_mlp_layer_units,
                                text_model, text_model_setting, code_model, code_model_setting, query_model,
                                query_model_setting, text_blocks, text_block_actual_lengths, code_blocks,
                                code_block_actual_lengths, queries, query_actual_lengths, block_sequence_actual_lengths,
                                keep_prob, selected_indices, dtype=tf.float32, random_seed=1):
  """ The hierarchical neural network.
  Args:

  Returns:
    code_labels: a tensor of shape [batch_size, 2].
    sentence_token_level_outputs: a len(text_blocks)+len(code_blocks) length list of tensors of shape [bs, block_size].
    block_rnn_outputs: a len(text_blocks)+len(code_blocks) length list of tensors of shape [bs, block_size*2]."""

  if text_model_setting.type != 0:
    if HNN_variant == 1:
      block_size, input_size = block_setting
    elif HNN_variant == 2:
      num_mlp_layers, mlp_layer_units = block_setting

  with tf.variable_scope("hierarchical_neural_network"):

    sentence_token_level_outputs = []
      # a list of [batch_size, block_size] Tensors, the outputs of the word_level rnn.
    summaries = [] # a list of summaries

    with tf.variable_scope("token_level_encoder"):
      # query
      if query_model_setting.type != 0 and query_model_setting == text_model_setting:
        query_rnn_scope = "text_block"
      else:
        query_rnn_scope = "query"

      with tf.variable_scope(query_rnn_scope):
        queries_semantic_size = query_model_setting.sem_size
        if query_model_setting.type == 0:
          queries_semantic = None
        else:
          queries_semantic = query_model.RNN_semantic(queries, query_actual_lengths)

        if query_model_setting.type != 0: # record
          with tf.name_scope("query"):
            summaries.extend(symmetric_variable_summaries(queries_semantic))

      if text_blocks is not None:
        block_length = len(text_blocks) * 2 - 1
      else:
        block_length = len(code_blocks) * 2 + 1

      text_semantic_size = text_model_setting.sem_size
      if text_model_setting.type !=0 and HNN_variant == 1:
        assert text_semantic_size == input_size

      for block_idx in range(block_length):

        if block_idx % 2 == 0:
          # text block
          if text_model_setting.type == 1:
            text_block_idx = int(block_idx / 2)
            with tf.variable_scope("text_block"):
              if query_rnn_scope == "text_block" or text_block_idx > 0:
                tf.get_variable_scope().reuse_variables()
              text_semantic =\
                text_model.RNN_semantic(text_blocks[text_block_idx], text_block_actual_lengths[text_block_idx])

              with tf.name_scope("text_block"):
                summaries.extend(symmetric_variable_summaries(text_semantic))
            sentence_token_level_outputs.append(text_semantic)

        else:
          # code block
          with tf.variable_scope("code_block"):
            code_block_idx = int(block_idx / 2)
            if code_block_idx > 0:
              tf.get_variable_scope().reuse_variables()

            if code_model_setting.type == 0:
              if HNN_variant == 1:
                batch_size = text_blocks[0][0].get_shape()[0].value
                code_semantic_vec = tf.get_variable("code_semantic_vector", shape=[input_size])
                code_semantic = tf.tile(tf.expand_dims(code_semantic_vec, dim=0), [batch_size, 1])
                code_semantic_size = input_size
              elif HNN_variant == 2:
                code_semantic = None
                code_semantic_size = 0

            else:
              code_semantic = code_model.RNN_semantic(code_blocks[code_block_idx],
                                                      code_block_actual_lengths[code_block_idx])
              code_semantic_size = code_model_setting.sem_size

              if queries_semantic is not None and code_semantic is not None:
                code_semantic = tf.concat([queries_semantic, code_semantic], axis=1)  # [bs, block_size*2]
                code_semantic_size = queries_semantic_size + code_semantic_size

              # MLP layers
              if code_num_mlp_layers > 0:
                code_semantic, code_semantic_summaries = MLP(
                  code_num_mlp_layers, code_mlp_layer_units, code_semantic, code_semantic_size,
                  "qc_feedforward_mlp", keep_prob=keep_prob, random_seed=random_seed)
                code_semantic_size = code_mlp_layer_units[-1]
                summaries.extend(code_semantic_summaries)
              else:
                # no MLP layer
                summaries.extend(symmetric_variable_summaries(code_semantic))

            sentence_token_level_outputs.append(code_semantic)

    if text_model_setting.type != 0:
      with tf.variable_scope("block_level_encoder"):
        if HNN_variant == 1: # RNN on block level
          assert code_semantic_size # should not be empty

          with tf.variable_scope("block_level_rnn"):
            with tf.variable_scope("GRU_cell", initializer=tf.random_uniform_initializer(
                    (-1) * np.sqrt(6.0 / (block_size + input_size)),
                np.sqrt(6.0 / (block_size + input_size)), dtype=tf.float32)):
              block_level_cell = tf.contrib.rnn.GRUCell(block_size)
              # if keep_prob < 1.0:
              #   block_level_cell = tf.contrib.rnn.DropoutWrapper(block_level_cell, input_keep_prob=keep_prob,
              #                                                    seed=random_seed)

            if keep_prob < 1:
              sentence_token_level_outputs = [tf.nn.dropout(item, keep_prob=keep_prob, seed=random_seed)
                                              for item in sentence_token_level_outputs]

            _sentence_token_level_outputs = tf.stack(sentence_token_level_outputs, axis=0)

            # block level RNN
            block_rnn_outputs, block_rnn_states = tf.nn.bidirectional_dynamic_rnn(
              block_level_cell, block_level_cell, _sentence_token_level_outputs,
              sequence_length=block_sequence_actual_lengths, time_major=True, dtype=tf.float32)
            block_rnn_outputs = tf.unstack(tf.concat(block_rnn_outputs, axis=2), axis=0) # a list of [batch_size, 2*block_size]
            block_level_output_size = block_size * 2

          # # pick the outputs for code snippets that we are interested in
          # _block_rnn_outputs = tf.stack(block_rnn_outputs,
          #                               axis=1)  # [bs, len(text_blocks)+len(code_blocks), block_level_output_size]
          # batch_size = text_blocks[0][0].get_shape()[0].value
          # first_dims = tf.expand_dims(tf.constant(range(batch_size), dtype=tf.int32), 1)
          # second_dims = tf.expand_dims(selected_indices, 1)
          # selected = tf.concat([first_dims, second_dims], axis=1)
          # selected_code_outputs = tf.gather_nd(_block_rnn_outputs, selected)  # [bs, block_level_output_size]

          # pick the code vec when there's only one code snippet
          assert len(block_rnn_outputs) == 3
          block_level_code_output = block_rnn_outputs[1]

        elif HNN_variant == 2: # MLP on block level
          with tf.variable_scope("block_level_mlp"):
            if code_model_setting.type == 0:
              concat_ff_input = tf.concat([sentence_token_level_outputs[0], sentence_token_level_outputs[2]], axis=1)
              concat_ff_input_size = text_semantic_size * 2
            else:
              concat_ff_input = tf.concat(sentence_token_level_outputs, axis=1)
              concat_ff_input_size = text_semantic_size * 2 + code_semantic_size

            # MLP
            block_level_code_output, block_level_summaries = MLP(
              num_mlp_layers, mlp_layer_units, concat_ff_input, concat_ff_input_size, "block_level_feedforward_mlp",
              keep_prob=keep_prob, random_seed=random_seed)
            block_level_output_size = mlp_layer_units[-1]
            summaries.extend(block_level_summaries)

    else:
      # NOTE: you can use these lines only for modeling one code snippet (instead of the entire post)
      assert len(sentence_token_level_outputs) == 1
      block_level_code_output = sentence_token_level_outputs[0] # [bs, code_semantic_size]
      block_level_output_size = code_semantic_size

    # record
    summaries.extend(variable_summaries(block_level_code_output, metrics=["mean-stddev"]))

    # dropout for the input of the prediction layer
    if keep_prob < 1.0:  # dropout
      block_level_code_output = tf.nn.dropout(block_level_code_output, keep_prob, seed=random_seed)

    # prediction layer
    with tf.variable_scope("code_label_prediction"):
      project_weight = tf.get_variable("weight", shape=[block_level_output_size, 2],
                                       initializer=tf.random_uniform_initializer(
                                         (-1) * np.sqrt(6.0 / block_level_output_size),
                                         np.sqrt(6.0 / block_level_output_size),
                                         dtype=dtype
                                       ))
      project_bias = tf.get_variable("bias", shape=[2], initializer=tf.zeros_initializer())
      code_labels = tf.matmul(block_level_code_output, project_weight) + project_bias

    return code_labels, sentence_token_level_outputs, summaries


def example_loss(logit, target):
  batch_size = logit.get_shape()[0].value
  with tf.name_scope("loss"):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logit)
    loss = tf.reduce_sum(crossent) / tf.cast(batch_size, tf.float32)

  return loss


def sequence_loss(logits, targets, target_weights=None, scope=None):
  """ Compute the loss function of the code label prediction.

  Args:
    logits: a list of tensors of shape [batch_size, 2], the predicted label.
    targets: a list of tensors of shape [batch_size], the gold standard label.
    target_weights: a list of tensors of shape [batch_size], the target weights for 
      evaluation.
    num_classes: the number of classes in the task.

  Returns:
    averaged_loss: tf.float32, the loss averaged on weights and batch size."""

  if len(logits) != len(targets):
    raise ValueError("Invalid argument size.")

  batch_size = targets[0].get_shape()[0].value

  with tf.name_scope(scope or "sequence_loss"):
    losses = []
    for i in xrange(len(logits)):
      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets[i], logits=logits[i])
      if target_weights is not None:
        losses.append(crossent * target_weights[i])
      else:
        losses.append(crossent)
    
    loss = tf.add_n(losses) #[batch_size]

    loss = tf.reduce_sum(loss)/tf.cast(batch_size, tf.float32)
    
  return loss


def model_with_buckets(buckets, HNN_variant, block_setting, code_num_mlp_layers, code_mlp_layer_units,
                       text_model, text_model_setting, code_model,
                       code_model_setting, query_model, query_model_setting,
                       text_blocks, text_block_actual_lengths, code_blocks, code_block_actual_lengths,
                       queries, query_actual_lengths, code_label_targets, selected_indices,
                       block_sequence_actual_lengths, keep_prob,
                       bool_has_attention=False, dtype=tf.float32, random_seed=1):
  """ Create a HNN model with buckets.
  Args:
    block_setting: when HNN_variant = 1, it contains [block_size, input_size];
                   when HNN_variant = 2, it contains [num_mlp_layers, mlp_layer_units (a list of int)].
    code_num_mlp_layers: the number of MLP layers for code encoding.
    code_mlp_layer_units: a list of integers, the unit size for each layer in code token-level MLP.
    text_model: the text model, "RNN" or "None".
    text_model_setting: the text model setting defined in config.
    code_model: the code model, "RNN" or "None".
    code_model_setting: the code model setting defined in config.
    query_model: the query model, "RNN" or "None".
    query_model_setting: the query model setting defined in config.
    text_blocks: a list of text blocks, each is a list of batch-sized list of integers (word id).
    text_block_actual_lengths: a list of batch-sized lists of integers (actual length of each text block).
    code_blocks: similar to text_blocks.
    code_block_actual_lengths: similar to text_block_actual_lengths.
    queries: a list of batch-sized integers (word ids).
    query_actual_lengths: a batch-sized list of integers (actual query length).
    code_label_targets: a batch-sized list of integers (code label).
    selected_indices: a batch-sized list of integers (code position index).
    block_sequence_actual_lengths: a batch-sized list of integers (actual length of each block sequence).
    keep_prob: keep probability for dropout.
    bool_has_attention: Please set to False for now.
    dtype: float32.

  Returns:
    outputs: a len(buckets) length list of tensors of shape [batch_size, 2].
    losses: a len(buckets) length list of tf.float32 scalars, the averaged loss."""


  with tf.variable_scope("model_with_buckets"):
    predictions = [] # length = len(buckets_text) * len(buckets_code)
    losses = []

    # for debugging
    token_level_outputs = []
    # block_level_outputs = []
    summaries = []


    for i in xrange(len(buckets)):
      if i > 0:
        tf.get_variable_scope().reuse_variables()

      text_block_length, text_word_length, query_word_length, code_token_length = buckets[i]

      # text blocks
      if text_model_setting.type == 0:
        bucket_text_blocks = None
        bucket_text_block_actual_lengths = None
      else:
        bucket_text_blocks = [text_blocks[idx_sent][:text_word_length] for idx_sent in xrange(text_block_length)]
        bucket_text_block_actual_lengths = text_block_actual_lengths[:text_block_length]

      # code blocks
      if code_model_setting.type == 0:
        bucket_code_blocks = None
        bucket_code_block_actual_lengths = None
      else:
        bucket_code_blocks = [code_blocks[idx_sent][:code_token_length] for idx_sent in xrange(text_block_length - 1)]
        bucket_code_block_actual_lengths = code_block_actual_lengths[:text_block_length - 1]

      # queries
      if query_model_setting.type == 0:
        bucket_queries = None
      else:
        bucket_queries = queries[:query_word_length]

      bucket_code_labels, bucket_token_level_outputs, bucket_summaries =\
        hierarchical_neural_network(
          HNN_variant, block_setting, code_num_mlp_layers, code_mlp_layer_units,
          text_model, text_model_setting, code_model, code_model_setting, query_model, query_model_setting,
          bucket_text_blocks, bucket_text_block_actual_lengths, bucket_code_blocks, bucket_code_block_actual_lengths,
          bucket_queries, query_actual_lengths, block_sequence_actual_lengths,
          keep_prob, selected_indices, dtype=dtype, random_seed=random_seed)

      predictions.append(bucket_code_labels)
      token_level_outputs.append(bucket_token_level_outputs)
      summaries.append(bucket_summaries)

      bucket_loss = example_loss(bucket_code_labels, code_label_targets)
      losses.append(bucket_loss)

  return losses, predictions, token_level_outputs, summaries



