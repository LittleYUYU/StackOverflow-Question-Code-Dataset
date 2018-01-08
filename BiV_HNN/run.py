# run.py
"""Run the experiment. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import random
import os
import pickle
import numpy as np
import math
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from scipy.sparse import csr_matrix
import gc

from model import HNNModel
from data_utils import *

import pdb

tf.app.flags.DEFINE_boolean("train", False, "Set to True if running to train.")
tf.app.flags.DEFINE_boolean("test", False, "Set to True if running to test.")
tf.app.flags.DEFINE_boolean("self_test", False, "Set to True if running the toy examples.")

tf.app.flags.DEFINE_integer("train_setting", 1, "Set the training set setting.\
                            1 ==> python, 2 ==> sql")
tf.app.flags.DEFINE_integer("context_setting", 1, "Set the context setting.\
                            1 ==> partialcontext_shared_text_vocab")
tf.app.flags.DEFINE_integer("HNN_variant", 1, "Set HNN variant for different model setting."
                            "1 ==> feeding s1, c1, s2 and etc into an RNN. Args: block_size, input_size."
                            "2 ==> concatenating s1, c(if any), s2 and feeding it into MLP. "
                                              "Args: num_mlp_layers, mlp_layer_units.")
# if HNN_variant = 1:
tf.app.flags.DEFINE_integer("block_size", 128, "Block-level GRU cell dimension.")
tf.app.flags.DEFINE_integer("input_size", 128, "Block-level GRU input size.")

# if HNN_variant = 2:
tf.app.flags.DEFINE_integer("num_mlp_layers", 1, "Number of MLP layers in block level."
                            "Note: the MLP doesn't include the prediction layer, which is a softmax(wx+b).")
tf.app.flags.DEFINE_string("mlp_layer_units", "128", "Unit size of each layer in block-level MLP.")

# code configuration
tf.app.flags.DEFINE_integer("code_num_mlp_layers", 1, "Number of MLP layers in token-level code encoder."
                            "Note: the MLP doesn't include the prediction layer, which is a softmax(wx+b).")
tf.app.flags.DEFINE_string("code_mlp_layer_units", "128", "Unit size of each layer in token-level MLP for code."
                           "Note: when HNN_variant=1 and code_model != 0, the last number should be the same as input_size.")

tf.app.flags.DEFINE_integer("text_model", 1, "Set the model setting for text blocks."
                                             "0 ==> no text model, 1 ==> GRU-RNN.")
tf.app.flags.DEFINE_integer("code_model", 1, "Set the model setting for code blocks."
                                             "0 ==> no code model, 1 ==> GRU-RNN.")
tf.app.flags.DEFINE_integer("query_model", 1, "Set the model setting for query."
                                              "0 ==> no query model, 1 ==> GRU-RNN.")
tf.app.flags.DEFINE_string("text_model_setting", None, "Set the text model configuration.")
tf.app.flags.DEFINE_string("code_model_setting", None, "Set the code model configuration.")
tf.app.flags.DEFINE_string("query_model_setting", None, "Set the query model configuration.")

# hyper-parameter setting
tf.app.flags.DEFINE_float("keep_prob", 1.0, "Set the keep probability for dropout.")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Set the learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size for training.")
tf.app.flags.DEFINE_integer("random_seed", 1, "Set the random seed for initialization.")

FLAGS = tf.app.flags.FLAGS


train_settings = ["python", "sql"]
context_settings = ["partialcontext_shared_text_vocab"]

# path
train_path = os.path.join("../data/data_hnn", train_settings[FLAGS.train_setting-1],
  "train/data_%s_in_buckets.pickle" % context_settings[FLAGS.context_setting-1])
dev_path = os.path.join("../data/data_hnn", train_settings[FLAGS.train_setting-1],
  "valid/data_%s_in_buckets.pickle" % context_settings[FLAGS.context_setting-1])
test_path = os.path.join("../data/data_hnn", train_settings[FLAGS.train_setting-1],
  "test/data_%s_in_buckets.pickle" % context_settings[FLAGS.context_setting-1])


# different checkpoint path for each setting
# text
if FLAGS.text_model != 0:
  if FLAGS.HNN_variant == 1:
    setting_record = "RNN_block%d_input%d" % (FLAGS.block_size, FLAGS.input_size)
  elif FLAGS.HNN_variant == 2:
    setting_record = "FF_mlp%d_units%s" % (FLAGS.num_mlp_layers, FLAGS.mlp_layer_units)
  else:
    raise Exception("Invalid argument HNN_variant!")
else:
  setting_record = ""
# code
if FLAGS.code_model != 0:
  setting_record += "_code_mlp%d_units%s" % (FLAGS.code_num_mlp_layers, FLAGS.code_mlp_layer_units)

checkpoint_path = os.path.join("../data/data_hnn", train_settings[FLAGS.train_setting-1],
  "checkpoint/ckpt_data_%s_%s_keepprob%.3f_lr%.3f_bs%d_seed%d_l2"
  "_text%d_setting%s_code%d_setting%s_query%d_setting%s" % (
    context_settings[FLAGS.context_setting-1], setting_record,
    FLAGS.keep_prob, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.random_seed,
    FLAGS.text_model, FLAGS.text_model_setting,
    FLAGS.code_model, FLAGS.code_model_setting,
    FLAGS.query_model, FLAGS.query_model_setting))


print("Loading from\ntrain_path: %s\ndev_path: %s"
  " \ntest_path: %s\ncheckpoint_path: %s"
  "\n\n" % (
  train_path, dev_path, test_path, checkpoint_path))

# buckets: (text block number, word token number, query token number, code token number)
buckets_train = buckets_dev = buckets_test = [(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)]
print("Buckets_text:\ntrain: %s\ndev: %s\ntest: %s.\n" % (
  str(buckets_train), str(buckets_dev), str(buckets_test)))


class DummySetting(object):
  def __init__(self):
    self.type = 0
    self.sem_size = 0

  def __str__(self):
    return str(self.__dict__)

  def __eq__(self, other):
    return self.__dict__ == other.__dict__

  def __ne__(self, other):
    return self.__dict__ != other.__dict__


class RNNSetting(object):
  def __init__(self, attr, configure_string):
    self.type = 1
    configures = configure_string.split("-")
    self.cell_size, self.input_size, self.word_vocab_size,\
      self.bool_has_attention, self.bool_pretrain_emb, self.bool_pretrain_model,\
      self.bool_bidirection = map(int, configures)
    self.sem_size = self.cell_size * 2 if self.bool_bidirection else self.cell_size

    self.pretrain_emb_path = os.path.join("../data/data_hnn/%s/" % train_settings[FLAGS.train_setting - 1],
      "train/rnn_partialcontext_word_embedding_%s_%d.pickle" % (attr, self.input_size))


  def __str__(self):
    return str(self.__dict__)

  def __eq__(self, other):
    return self.__dict__ == other.__dict__

  def __ne__(self, other):
    return self.__dict__ != other.__dict__


## training configurations
class SelfTestConfig(object):
  HNN_variant = 1
  if FLAGS.text_model != 0:
    if HNN_variant == 1:
      block_size = 16
      input_size = 16
    elif HNN_variant == 2:
      num_mlp_layers = 2
      mlp_layer_units = [int(unit) for unit in "50-10".split("-")]
  if FLAGS.code_model != 0:
    code_num_mlp_layers = 0
    code_mlp_layer_units = None
    if FLAGS.text_model != 0 and HNN_variant == 1 and code_num_mlp_layers > 0:
      assert input_size == code_mlp_layer_units[-1]

  max_epoch = 10
  batch_size = 2
  learning_rate = 0.5
  keep_prob = 0.5
  init_scale = 0.08
  random_seed = FLAGS.random_seed

  def __init__(self):
    # self.text_model = DummySetting()
    self.text_model = RNNSetting("text", "8-8-6-0-0-0-1")
    # self.code_model = DummySetting()
    self.code_model = RNNSetting("code", "8-8-7-0-0-0-1")
    self.query_model = DummySetting()
    # self.query_model = RNNSetting("text", "8-8-6-0-0-0-1")


class TrainConfig(object):
  HNN_variant = FLAGS.HNN_variant
  if FLAGS.text_model != 0:
    if HNN_variant == 1:
      block_size = FLAGS.block_size
      input_size = FLAGS.input_size
    elif HNN_variant == 2:
      num_mlp_layers = FLAGS.num_mlp_layers
      if num_mlp_layers > 0:
        mlp_layer_units = [int(unit) for unit in FLAGS.mlp_layer_units.split("-")]
      else:
        mlp_layer_units = None
  if FLAGS.code_model != 0:
    code_num_mlp_layers = FLAGS.code_num_mlp_layers
    if code_num_mlp_layers > 0:
      code_mlp_layer_units = [int(unit) for unit in FLAGS.code_mlp_layer_units.split("-")]
    else:
      code_mlp_layer_units = None
    if FLAGS.text_model != 0 and HNN_variant == 1 and code_num_mlp_layers > 0:
      assert input_size == code_mlp_layer_units[-1]

  max_epoch = 70
  batch_size = FLAGS.batch_size
  learning_rate = FLAGS.learning_rate
  keep_prob = FLAGS.keep_prob
  init_scale = 0.08
  random_seed = FLAGS.random_seed

  def __init__(self):
    # text model
    self.text_model = None
    if FLAGS.text_model == 0:
      self.text_model = DummySetting()
    elif FLAGS.text_model == 1:
      self.text_model = RNNSetting("text", FLAGS.text_model_setting)
    else:
      raise Exception("Invalid text model!")

    # code model
    self.code_model = None
    if FLAGS.code_model == 0:
      self.code_model = DummySetting()
    elif FLAGS.code_model == 1:
      self.code_model = RNNSetting("code", FLAGS.code_model_setting)
    else:
      raise Exception("Invalid code model!")

    # query model
    self.query_model = None
    if FLAGS.query_model == 0:
      self.query_model = DummySetting()
    elif FLAGS.query_model == 1:
      self.query_model = RNNSetting("text", FLAGS.query_model_setting)
    else:
      raise Exception("Invalid query model!")


class TestConfig(object):
  HNN_variant = FLAGS.HNN_variant
  if FLAGS.text_model != 0:
    if HNN_variant == 1:
      block_size = FLAGS.block_size
      input_size = FLAGS.input_size
    elif HNN_variant == 2:
      num_mlp_layers = FLAGS.num_mlp_layers
      if num_mlp_layers > 0:
        mlp_layer_units = [int(unit) for unit in FLAGS.mlp_layer_units.split("-")]
      else:
        mlp_layer_units = None
  if FLAGS.code_model != 0:
    code_num_mlp_layers = FLAGS.code_num_mlp_layers
    if code_num_mlp_layers > 0:
      code_mlp_layer_units = [int(unit) for unit in FLAGS.code_mlp_layer_units.split("-")]
    else:
      code_mlp_layer_units = None
    if FLAGS.text_model != 0 and HNN_variant == 1 and code_num_mlp_layers > 0:
      assert input_size == code_mlp_layer_units[-1]

  max_epoch = 0
  HNN_variant = FLAGS.HNN_variant
  batch_size = 100
  learning_rate = 0.0
  keep_prob = 1.0
  init_scale = 0.08
  random_seed = FLAGS.random_seed

  def __init__(self):
    # text model
    self.text_model = None
    if FLAGS.text_model == 0:
      self.text_model = DummySetting()
    elif FLAGS.text_model == 1:
      self.text_model = RNNSetting("text", FLAGS.text_model_setting)
    else:
      raise Exception("Invalid text model!")

    # code model
    self.code_model = None
    if FLAGS.code_model == 0:
      self.code_model = DummySetting()
    elif FLAGS.code_model == 1:
      self.code_model = RNNSetting("code", FLAGS.code_model_setting)
    else:
      raise Exception("Invalid code model!")

    # query model
    self.query_model = None
    if FLAGS.query_model == 0:
      self.query_model = DummySetting()
    elif FLAGS.query_model == 1:
      self.query_model = RNNSetting("text", FLAGS.query_model_setting)
    else:
      raise Exception("Invalid query model!")


def create_model(session, buckets, config,
                 text_embedding=None, code_embedding=None, query_embedding=None,
                 forward_only=False):
  model = HNNModel(buckets, config, forward_only=forward_only)

  if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  if FLAGS.test: assert ckpt
  if ckpt:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    saver = tf.train.Saver()
    saver.restore(session, ckpt.model_checkpoint_path)

  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())

    if config.text_model.type == 1 and config.text_model.bool_pretrain_emb:  # text block
      assert text_embedding is not None
      print("Load pretrained text embedding.")
      session.run(model.text_word_embedding.assign(text_embedding))
    if config.code_model.type == 1 and config.code_model.bool_pretrain_emb:  # code block
      assert code_embedding is not None
      print("Load pretrained code embedding.")
      session.run(model.code_token_embedding.assign(code_embedding))
    # load textual word embedding for the query model only when text_model=0.
    if config.text_model.type == 0 and config.query_model.type == 1 and config.query_model.bool_pretrain_emb:
      assert query_embedding is not None
      print("Load pretrained query embedding.")
      session.run(model.query_word_embedding.assign(query_embedding))

  sys.stdout.flush()
  return model


def get_embedding(config):
  text_embedding = None
  code_embedding = None
  query_embedding = None
  # text blocks
  if config.text_model.type == 1 and config.text_model.bool_pretrain_emb:
    text_embedding = pickle.load(open(config.text_model.pretrain_emb_path, "rb"))
    if isinstance(text_embedding, csr_matrix):
      text_embedding = text_embedding.toarray()
    print("Loading embedding for text blocks...size %s." % str(text_embedding.shape))

  # code block
  if config.code_model.type == 1 and config.code_model.bool_pretrain_emb:
    code_embedding = pickle.load(open(config.code_model.pretrain_emb_path, "rb"))
    if isinstance(code_embedding, csr_matrix):
      code_embedding = code_embedding.toarray()
    print("Loading embedding for code blocks...size %s." % str(code_embedding.shape))

  # query block
  if config.text_model.type == 0: # otherwise, no need to load the same pretrained embedding
    if config.query_model.type == 1 and config.query_model.bool_pretrain_emb:
      query_embedding = pickle.load(open(config.query_model.pretrain_emb_path, "rb"))
      if isinstance(query_embedding, csr_matrix):
        query_embedding = query_embedding.toarray()
      print("Loading embedding for query...size %s." % str(query_embedding.shape))
  else:
    query_embedding = text_embedding
    print("Apply text embedding for query...size %s." % str(query_embedding.shape))


  return text_embedding, code_embedding, query_embedding


def train():
  # load data
  print("\nLoading data...")
  with open(train_path, "rb") as f:
    train_data = pickle.load(f)
  print("Number of training data: %d" % sum([len(bucket_data) for bucket_data in train_data]))
  with open(dev_path, "rb") as f:
    dev_data = pickle.load(f)
  with open(test_path, "rb") as f:
    test_data = pickle.load(f)

  # check the bucket size
  train_bucket_size = [len(train_data[b]) for b in xrange(len(buckets_train))]
  train_total_size = float(sum(train_bucket_size))
  train_bucket_scale = [sum(train_bucket_size[:i + 1]) / train_total_size
                        for i in xrange(len(train_bucket_size))]

  # set the random seed
  tf.set_random_seed(FLAGS.random_seed)
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)

  # config
  config = TrainConfig()
  dev_config = TestConfig()
  max_batch = int(train_total_size // config.batch_size) + 1

  text_embedding, code_embedding, query_embedding = get_embedding(config)
  sys.stdout.flush()

  # run session
  with tf.Session() as sess:

    initializer = tf.random_uniform_initializer(
      -config.init_scale, config.init_scale)
    
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      model = create_model(sess, buckets_train, config,
                           text_embedding, code_embedding, query_embedding, forward_only=False)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      dev_model = HNNModel(buckets_dev, dev_config, forward_only=True)

    train_record_path = os.path.join(checkpoint_path, "train_record")
    if not os.path.exists(train_record_path):
      os.makedirs(train_record_path)
    # train_writer = tf.summary.FileWriter(train_record_path)

    sys.stdout.flush()

    epoch_train_losses_during_training = []
    epoch_dev_losses, epoch_dev_precision, epoch_dev_recall,\
      epoch_dev_f1, epoch_dev_accuracy = [], [], [], [], []
    epoch_test_losses, epoch_test_precision, epoch_test_recall,\
      epoch_test_f1, epoch_test_accuracy = [], [], [], [], []
    epoch_train_losses, epoch_train_precision, epoch_train_recall,\
      epoch_train_f1, epoch_train_accuracy = [], [], [], [], []
    previous_dev_f1 = float(0)
    count_low_f1 = 0 # for early stopping

    gc.enable() # enabling garbage collection. (untested)

    for epoch_idx in xrange(1, config.max_epoch+1):
      # for each epoch

      print("\nTraining...")      

      train_losses = []
      for batch_idx in xrange(max_batch):
        # for each batch training
        random_number = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_bucket_scale)) 
          if train_bucket_scale[i] >= random_number])

        text_blocks, text_block_actual_lengths, code_blocks, code_block_actual_lengths, \
        queries, query_actual_lengths, code_label_targets, code_selected_indices, block_sequence_actual_length = \
          model.get_batch(train_data, bucket_id, text_embedding=text_embedding,
                          code_embedding=code_embedding, query_embedding=query_embedding)
        loss, prediction = \
          model.step(sess, text_blocks, text_block_actual_lengths, code_blocks, code_block_actual_lengths,
                    queries, query_actual_lengths, code_label_targets, code_selected_indices,
                    block_sequence_actual_length, bucket_id, forward_only=False)
        # train_writer.add_summary(summary, global_step=model._global_step.eval())
        train_losses.append(loss)

      train_loss = float(sum(train_losses)) / max_batch
      print("At %d-th epoch, %.3f loss on training data." % (epoch_idx, train_loss))
      epoch_train_losses_during_training.append(train_loss)

      if train_loss <= 1e-4: # early stop by measuring training loss
        break

      # test on dev data
      if epoch_idx:
        print("Training data...")
        train_loss, train_precision, train_recall, train_f1, train_accuracy =\
          test_and_evaluate(sess, dev_model, train_data, buckets_train,
                            text_embedding=text_embedding, code_embedding=code_embedding,
                            query_embedding=query_embedding)
        epoch_train_losses.append(train_loss)
        epoch_train_precision.append(train_precision)
        epoch_train_recall.append(train_recall)
        epoch_train_f1.append(train_f1)
        epoch_train_accuracy.append(train_accuracy)

        print("Validation...")
        dev_loss, dev_precision, dev_recall, dev_f1, dev_accuracy =\
          test_and_evaluate(sess, dev_model, dev_data, buckets_dev,
                            text_embedding=text_embedding, code_embedding=code_embedding,
                            query_embedding=query_embedding)
        epoch_dev_losses.append(dev_loss)
        epoch_dev_precision.append(dev_precision)
        epoch_dev_recall.append(dev_recall)
        epoch_dev_f1.append(dev_f1)
        epoch_dev_accuracy.append(dev_accuracy)

        print("Test...")
        test_loss, test_precision, test_recall, test_f1, test_accuracy = \
          test_and_evaluate(sess, dev_model, test_data, buckets_test,
                            text_embedding=text_embedding, code_embedding=code_embedding,
                            query_embedding=query_embedding)
        epoch_test_losses.append(test_loss)
        epoch_test_precision.append(test_precision)
        epoch_test_recall.append(test_recall)
        epoch_test_f1.append(test_f1)
        epoch_test_accuracy.append(test_accuracy)

        if dev_f1 > previous_dev_f1:
          checkpoint_save_path = os.path.join(checkpoint_path, "ckpt")
          print("Saving the model params to %s.\n" % checkpoint_save_path)
          model.saver.save(sess, checkpoint_save_path, global_step=model._global_step, write_meta_graph=False)
          previous_dev_f1 = dev_f1
          count_low_f1 = 0
          print("Update max f1: %.3f" % previous_dev_f1); sys.stdout.flush()

        else:
          print("Best f1 on dev set: %.3f.\nCurrent f1 on dev set: %.3f." % (previous_dev_f1, dev_f1))
          if previous_dev_f1 - dev_f1 > 0.03: # early stopping
            count_low_f1 += 1
            print("Early stopping count %d." % count_low_f1)
            if count_low_f1 == 5:
              print("Early stop!")
              break

      sys.stdout.flush() 

    # train_writer.close()

    max_idx = np.argmax(epoch_dev_f1)
    print("Max dev f1 is achieved by %d-th epoch: precision=%.3f, recall=%.3f, f1=%.3f, accuracy=%.3f" % (
      max_idx, epoch_dev_precision[max_idx], epoch_dev_recall[max_idx], epoch_dev_f1[max_idx],
      epoch_dev_accuracy[max_idx]))
    print("Corresponding test performance: precision=%.3f, recall=%.3f, f1=%.3f, accuracy=%.3f" % (
      epoch_test_precision[max_idx], epoch_test_recall[max_idx], epoch_test_f1[max_idx],
      epoch_test_accuracy[max_idx]))
    print("*"*10) # for formatting
    sys.stdout.flush()


def test_and_evaluate(sess, model, dataset, buckets,
                      text_embedding=None, code_embedding=None, query_embedding=None,
                      savepath=None, bool_eval=True):
  tp, fn, tn, fp = 0, 0, 0, 0
  
  losses = []
  predict_collections = []

  for bucket_id in xrange(len(buckets)):
    if len(dataset[bucket_id]) == 0:
      print("Bucket %d is empty. " % bucket_id)
      continue
    else:
      print("Bucket %d: %d examples." % (bucket_id, len(dataset[bucket_id])))
    sys.stdout.flush()

    num_runs = len(dataset[bucket_id]) // model._batch_size + 1
    for run_idx in range(num_runs):
      if (run_idx + 1) * model._batch_size <= len(dataset[bucket_id]):
        data_batch = dataset[bucket_id][run_idx * model._batch_size: (run_idx + 1) * model._batch_size]
        num_pad = 0
      else:
        # the remaining data size is less than the batch size
        data_batch = dataset[bucket_id][run_idx * model._batch_size:]
        # repeatedly append the last example until full
        num_pad = model._batch_size - len(data_batch)
        data_batch.extend([dataset[bucket_id][-1]] * num_pad)
      assert len(data_batch) == model._batch_size

      # note that we test one example each time
      text_blocks, text_block_actual_lengths, code_blocks, code_block_actual_lengths, \
      queries, query_actual_lengths, code_label_targets, code_selected_indices, block_sequence_actual_length = \
          model.get_batch({bucket_id:data_batch}, bucket_id, text_embedding=text_embedding,
                          code_embedding=code_embedding, query_embedding=query_embedding, bool_shuffle=False)
      loss, prediction =\
          model.step(sess, text_blocks, text_block_actual_lengths, code_blocks, code_block_actual_lengths,
                     queries, query_actual_lengths, code_label_targets, code_selected_indices,
                     block_sequence_actual_length, bucket_id, forward_only=True)

      gold_batch = [item[-3] for item in data_batch][:model._batch_size - num_pad]
      pred_batch = [1 if item[1] > item[0] else 0 for item in prediction][:model._batch_size - num_pad]

      for data_idx, (gold, pred) in enumerate(zip(gold_batch, pred_batch)):
        if bool_eval:
          if gold == 1:
            if pred == gold:
              tp += 1
            else:
              fn += 1
          else:
            if pred == gold:
              tn += 1
            else:
              fp += 1
        predict_collections.append((data_batch[data_idx][0], (gold, prediction[data_idx]), bucket_id))

      losses.append(loss) # record the loss
  
  if bool_eval:
    averaged_loss = float(sum(losses))/len(losses)
    if tp+fp == 0:
      precision = 0.0
    else:
      precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    if precision+recall == 0:
      f1 = float(0)
    else:
      f1 = 2*precision*recall/(precision+recall)
    accuracy = float(tp+tn)/(tp+fn+tn+fp)
    print("Averaged loss=%.3f, precision=%.3f, recall=%.3f, f1=%.3f, accuracy=%.3f"%(
      averaged_loss, precision, recall, f1, accuracy))
    print("="*10) # for formatting
    print("%.3f,%.3f,%.3f,%.3f" % (precision, recall, f1, accuracy))
  else:
    averaged_loss, precision, recall, f1, accuracy = [0.0] * 5

  if savepath:
    with open(savepath, "wb") as f:
      pickle.dump(predict_collections, f)
    print("Save to %s..." % savepath)

  return averaged_loss, precision, recall, f1, accuracy


def test():

  bool_eval = True # NOTE: please set to False when there is no ground truth.

  data_path = test_path
  buckets = buckets_test
  unlabeled_data = pickle_load(data_path)

  save_path = os.path.join(checkpoint_path, "test_result.pickle")
  # save_path = None

  # config
  config = TestConfig()
  config.batch_size = 100
  text_embedding, code_embedding, query_embedding = get_embedding(config)

  with tf.Session() as sess:
    with tf.variable_scope("model"):
      test_model = create_model(sess, buckets, config,
                           text_embedding, code_embedding, query_embedding, forward_only=True)
    loss, precision, recall, f1, accuracy =\
      test_and_evaluate(sess, test_model, unlabeled_data, buckets, text_embedding=text_embedding,
                        code_embedding=code_embedding,
                        query_embedding=query_embedding, savepath=save_path, bool_eval=bool_eval)
    if bool_eval:
      print("Loss=%.3f, precision=%.3f, recall=%.3f, f1=%.3f, accuracy=%.3f" % (
        loss, precision, recall, f1, accuracy))


def self_test():
  """ Self-test. """
  config = SelfTestConfig()

  buckets = [(2, 3, 4, 3), (2, 5, 4, 3)]

  # set the random seed
  tf.set_random_seed(FLAGS.random_seed)
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)

  # fake embedding
  text_embedding = np.random.rand(6, 8)
  code_embedding = np.random.rand(7, 8)
  # query_embedding = np.random.rand(6, 8)

  # fake data
  data = {}
  data00 = ("00", [[1, 2, 0], [1, 2, 2]], [3, 3],
            [[1, 2, 6]], [3], [1, 2, 3, 4], 4, 1, 1, 3)
  data01 = ("01", [[1, 2, 0], [0, 0, 0]], [2, 0],
            [[1, 3, 6]], [3], [5, 4, 3, 0], 3, 0, 1, 2)
  data[0] = [data00, data01]
  data10 = ("10", [[1, 2, 3, 4, 0], [1, 3, 0, 0, 0]], [5, 3],
            [[1, 2, 6]], [3], [1, 2, 3, 4], 4, 1, 1, 3)
  data11 = ("11", [[1, 2, 0, 0, 0], [1, 4, 0, 0, 0]], [3, 3],
            [[1, 3, 6]], [3], [5, 4, 3, 0], 3, 0, 1, 2)
  data[1] = [data10, data11]

  max_batch = int((len(data[0]) + len(data[1])) // config.batch_size)

  with tf.Session() as sess:

    # model definition
    with tf.variable_scope("model", reuse=None):
      model = create_model(sess, buckets, config, forward_only=False)
      model.writer = tf.summary.FileWriter("./summary", graph=sess.graph)

    # run the session
    sess.run(tf.global_variables_initializer())

    for _ in xrange(config.max_epoch):
      print("="*10)
      for batch_idx in xrange(max_batch):
        print("Batch %d:" % batch_idx)
        bucket_id = batch_idx % 2
        text_blocks, text_block_actual_lengths, code_blocks, code_block_actual_lengths, \
        queries, query_actual_lengths, code_label_targets, code_selected_indices, block_sequence_actual_length = \
          model.get_batch(data, bucket_id, text_embedding=text_embedding,
                          code_embedding=code_embedding, query_embedding=text_embedding)
        loss, prediction = \
           model.step(sess, text_blocks, text_block_actual_lengths, code_blocks, code_block_actual_lengths,
                      queries, query_actual_lengths, code_label_targets, code_selected_indices,
                      block_sequence_actual_length, bucket_id, forward_only=False)
        print(loss)   



def main():
  if FLAGS.train:
    train()
  elif FLAGS.test:
    test()
  elif FLAGS.self_test:
    self_test()


if __name__ == "__main__":
  main()

  
  

