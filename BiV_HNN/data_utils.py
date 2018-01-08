# data_utils.py
""" data utils. """

import gc
import math
import pickle, random
import bs4
from bs4 import BeautifulSoup as bs
import re
import operator
from nltk.tokenize import sent_tokenize,word_tokenize, RegexpTokenizer
from nltk.stem import SnowballStemmer
import numpy as np
import sys
from scipy.spatial.distance import *

from utils import sortDictByValue

PAD = 0


def pickle_load(filename):
  print("Loading pickle file from %s..." % filename)
  with open(filename, "rb") as f:
    data = pickle.load(f)
  return data


def pickle_save(data, filename):
  print("Saving data to pickle file %s..." % filename)
  with open(filename, "wb") as f:
    pickle.dump(data, f, protocol=2)


def text_process(tstring):
  """ Tokenizing, stemming. Return a list of tokens. """
  # reg_tokenizer = RegexpTokenizer(r'\w+')
  stemmer = SnowballStemmer("english")
  tokens = [stemmer.stem(word) for sent in sent_tokenize(tstring)\
    for word in word_tokenize(sent)]
  return tokens


def get_HAN_data(pickledata, pickle_valid_qids):
  data = pickle_load(pickledata)

  if pickle_valid_qids:
    all_qids = pickle_load(pickle_valid_qids)

  else:
    all_qids = data.keys()

  count_failed = 0
  count_total = 0

  qid_text_code_codelabel_tuples = []
  qid_partial_text_code_codelabel_tuples = []
  
  for iid in all_qids:
    qid = iid
    text_code_blocks = data[qid]['answer_text_code_blocks']
    
    text, partial_text, code_blocks = [],[], []
    code_blocks_index = []
    for idx in range(len(text_code_blocks)):
      if "</code></pre>" in text_code_blocks[idx]:
        code_blocks.append(bs(text_code_blocks[idx], "html.parser").text)
        code_blocks_index.append(idx)
        if idx > 0 and (len(code_blocks_index) == 1 or code_blocks_index[-2] != idx - 1):
          partial_text.append(text_process(bs(text_code_blocks[idx-1], "html.parser").text))
          if len(code_blocks_index) == 1:
            l = [text_process(bs(i, "html.parser").text)
              for i in text_code_blocks[:idx]]
            text.append([iterm for subl in l for iterm in subl])
          else:
            l = [text_process(bs(i, "html.parser").text)
              for i in text_code_blocks[code_blocks_index[-2]+1:idx]]
            text.append([iterm for subl in l for iterm in subl])
        else:
          partial_text.append([])
          text.append([])
    # append the context for the last code snippet
    try:
      assert len(code_blocks_index) > 0
    except:
      print(iid)
      count_failed += 1
      continue

    last_code_block_index = code_blocks_index[-1]
    if last_code_block_index == len(text_code_blocks) - 1:
      partial_text.append([])
      text.append([])
    else:
      partial_text.append(text_process(bs(text_code_blocks[last_code_block_index+1], "html.parser").text))
      l = [text_process(bs(i,"html.parser").text)
        for i in text_code_blocks[last_code_block_index+1:]]
      text.append([iterm for subl in l for iterm in subl])

    labels = [None for _ in range(len(text) - 1)]
    assert(len(text) == len(partial_text) == len(labels) + 1 == len(code_blocks) + 1)

    qid_partial_text_code_codelabel_tuples.append((iid, partial_text, code_blocks, [int(l) if l is not None else None for l in labels]))
    qid_text_code_codelabel_tuples.append((iid, text, code_blocks, [int(l) if l is not None else None for l in labels]))

    count_total += 1
    if count_total % 500 == 0:
      print(count_total)
      sys.stdout.flush()
  
  print("In total: %d" % len(qid_text_code_codelabel_tuples))
  
  return qid_text_code_codelabel_tuples, qid_partial_text_code_codelabel_tuples


def get_vocab(picklefile, picklesave, min_threshold=2, valid_iids=None, valid_qids=None):
  """ Generate the vocabulary. 
  Form of data in picklefile:
  [(inputs1, targets1), (input2, targets2), ...]
  where
  inputs1 = [[w11,w12,...], [w21,w22,...], ...]
  and
  targets = [label1, label2, ...].

  Returns:
    vocab_size."""

  word_count = dict()

  data = pickle_load(picklefile)
  print(len(data))

  for qid, text, code, label in data:
    if valid_qids is None or qid in valid_qids:

      if valid_iids is None:
        for idx in range(len(text)):
          l = text[idx]
          for token in l:
            word_count[token] = word_count.get(token, 0) + 1

      else:
        for idx in range(len(code)):
          iid = (qid, idx)
          if iid in valid_iids:
            l1 = text[idx]
            l2 = text[idx + 1]
            for token in l1+l2:
              word_count[token] = word_count.get(token, 0) + 1

  sorted_word_count = sortDictByValue(word_count)
  print("Before cut: %d" % len(sorted_word_count))

  vocab = ["PAD"] + [word for word, count in sorted_word_count if count >= min_threshold]

  print("Vocab size: %d." % len(vocab))

  with open(picklesave, "wb") as f:
    pickle.dump(vocab, f)

  print("Save vocab in %s." % picklesave)

  return sorted_word_count, vocab, len(vocab)


def load_vocab(picklevocab):
  with open(picklevocab, "rb") as f:
    vocab = pickle.load(f)
  return len(vocab), vocab


def get_pretrained_word_vec(word_emb_dict, vocab, dim):
  vectors = []

  for word in vocab:
    if word == "PAD" or word == "##PAD##":
      vectors.append(np.zeros([dim]))
    # elif word == "#UNK#":
    #   vectors.append(word_emb_dict["<unk>"])
    else:
      vectors.append(word_emb_dict.get(word, word_emb_dict["<unk>"]))
  vectors = np.array(vectors)
  print("Shape: %s" % str(vectors.shape))

  return vectors


def data_to_symbols(picklefile, picklevocab, picklesave):
  """ Transform data to symbols based on the given vocab. """
  with open(picklefile, "rb") as f:
    data = pickle.load(f)
  vocab_size, vocab = load_vocab(picklevocab)
  print("vocabulary size: %d" % vocab_size)

  symbols_label_pair = []
  for iid, text, code, label in data:
    symbols = []
    for l in text:
      # tokens = [vocab.index(token) if token in vocab else UNK for token in l]
      tokens = [vocab.index(token) for token in l if token in vocab]
      symbols.append(tokens)
    symbols_label_pair.append((iid, symbols, code, label))

  with open(picklesave, "wb") as f:
    pickle.dump(symbols_label_pair, f)
  print("Save to %s" % picklesave)

  return symbols_label_pair


def single_symbols_to_rnn_symbols(picklesingle, picklevocab, picklernn):
  with open(picklevocab, "rb") as f:
    vocab = pickle.load(f)
  with open(picklesingle, "rb") as f:
    single = pickle.load(f)
  code_idx = len(vocab)
  print("code_idx: %d" % code_idx)
  new_data = []
  for iid, text, code, label in single:
    text_i = []
    text_i.extend(text[0])
    text_i.append(code_idx)
    text_i.extend(text[1])
    new_data.append((iid, text_i, code, label))
  with open(picklernn, "wb") as f:
    pickle.dump(new_data, f)


def get_HAN_bucket_data(buckets, picklefile, picklesave, truncate_on=False):
  """ Process data to fit different buckets.

  Data form:
  {bucket_id: [(iid(or a tuple of (iid1, index1)), inputs1, targets1),
    (iid(or a tuple of (iid2, index2)), inputs2, targets2), ...]}
  where
  inputs1 = [[w11, w12, ...], [w21, w22, ...], ...]
  and
  targets1 = [label1, label2, ...]. 
  
  Args:
    buckets: a list of tuples of int numbers."""

  with open(picklefile, "rb") as f:
    data = pickle.load(f)

  data_in_buckets = [[] for _ in range(len(buckets))]
  
  count_left = 0
  for iid_info, text, code, label in data:
    max_len = max([len(i) for i in text])
    if max_len > buckets[-1][1] or len(label)+1 > buckets[-1][0]:
      if truncate_on and len(label)+1 <= buckets[-1][0]:
        text = [l if len(l)<=buckets[-1][1] else l[:buckets[-1][1]] for l in text]
        max_len = buckets[-1][1]
      else:
        count_left += 1
        # print(iid_info, max_len, len(label))
        continue
    idx = min([i for i in range(len(buckets)) 
      if len(label)+1 <= buckets[i][0] and max_len <= buckets[i][1]])
    data_in_buckets[idx].append((iid_info, text, code, label))

  data_in_buckets_size = [len(_) for _ in data_in_buckets]

  print("Bucket: %s" % str(buckets))
  print("Data size: \n In total: %d.\nBucket size: %s." % (
    sum(data_in_buckets_size), str(data_in_buckets_size)))
  print("%d examples left.\n" % count_left)

  with open(picklesave, "wb") as f:
    pickle.dump(data_in_buckets, f, protocol=2)
  print("Save to %s." % picklesave)


def symbols_to_data(vocab, list_of_symbols):
  """ Transform the symbol to original word.
  Args:
    vocab: a list of words, the dictionary.
    list_of_symbols: a list of string lists. """
  list_of_sentences = []
  for symbols in list_of_symbols: # a list of symbols
    sentence = [vocab[i] for i in symbols]
    list_of_sentences.append(sentence)
  return list_of_sentences 


def load_embedding(picklepath):
  with open(picklepath, "rb") as f:
    embedding = pickle.load(f)
  print("Pretrained embedding matrix with shape: %s" % str(embedding.shape))
  return embedding


###############
## update 07/16
###############

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()


def get_new_hnn_format_single_code(qid_texts2ids_codes_labels, qid_to_code, qid_to_query,
                                   all_qid_to_idx=None, iid_to_label=None):

  iid_and_content = []

  for qid, texts2ids, codes, labels in qid_texts2ids_codes_labels:
    if all_qid_to_idx is None or qid in all_qid_to_idx:

      # query
      query = qid_to_query[qid]
      query_actual_length = len(query)

      for idx in all_qid_to_idx[qid]:
        iid = (qid, idx)

        # text blocks
        text_blocks = [texts2ids[idx], texts2ids[idx+1]]
        text_block_actual_lengths = [len(texts2ids[idx]), len(texts2ids[idx+1])]

        # code block
        if iid not in qid_to_code: # in case of failed code tokenization
          continue
        code2id = qid_to_code[iid]
        code_blocks = [code2id]
        code_block_actual_lengths = [len(code2id)]

        # code label
        code_label = labels[idx] or 0
        if iid_to_label:
          code_label = iid_to_label[iid]
        # code selection
        code_idx = 1

        post_actual_length = 3
        if text_block_actual_lengths[1] == 0:
          post_actual_length = 2

        iid_and_content.append([iid, text_blocks, text_block_actual_lengths, code_blocks, code_block_actual_lengths,
                               query, query_actual_length, code_label, code_idx, post_actual_length])

  return iid_and_content


def get_new_hnn_format_multiple_code(qid_texts2ids_codes_labels, qid_to_code, qid_to_query,
                                   all_qid_to_idx=None, iid_to_label=None):
  iid_and_content = []

  for qid, texts2ids, codes, labels in qid_texts2ids_codes_labels:
    if all_qid_to_idx is None or qid in all_qid_to_idx:

      # query
      query = qid_to_query[qid]
      query_actual_length = len(query)

      all_indices = all_qid_to_idx[qid]
      valid_indices = []
      if len(all_indices) == 1:
        valid_indices.append([0])
      else:
        for all_indices_idx, valid_idx in enumerate(all_indices):
          valid_indices.append([valid_idx])
          if len(all_indices[all_indices_idx:]) > 1 and all_indices[all_indices_idx+1] == valid_idx + 1:
            valid_indices.append([valid_idx, valid_idx+1])

      for indices in valid_indices:
        iid = (qid, indices)

        # text blocks
        text_blocks = [texts2ids[indices[0]:indices[0]+len(indices)]]
        text_block_actual_lengths = [len(item) for item in text_blocks]

        # code block
        code_blocks = [qid_to_code[(qid, idx)] for idx in indices]
        code_block_actual_lengths = [len(code_block) for code_block in code_blocks]
        assert len(code_blocks) == len(text_blocks) - 1

        # code label
        if iid_to_label:
          code_labels = [iid_to_label[(qid, idx)] for idx in indices]
        else:
          code_labels = [labels[idx] for idx in indices]
        # code selection
        code_indices = [2*i+1 for i in range(len(code_labels))]

        post_actual_length = len(text_blocks) + len(code_blocks)
        if text_block_actual_lengths[1] == 0:
          post_actual_length -= 1

        iid_and_content.append([iid, text_blocks, text_block_actual_lengths, code_blocks, code_block_actual_lengths,
                                query, query_actual_length, code_labels, code_indices, post_actual_length])

  return iid_and_content


def get_new_hnn_format_bucket(iid_data, buckets, truncate_on=False):
  """ The buckets should be a list of size-4 tuples: (#text blocks, #text words, #query words, #code tokens). """
  data_in_buckets = [[] for _ in range(len(buckets))]

  def padding_inputs(ids, bucket_size):
    if bucket_size < len(ids):
      padded_ids = ids[:bucket_size]
    else:
      padded_ids = ids + [0] * (bucket_size - len(ids))
    return padded_ids

  def padding_data_instance(instance, bucket):
    """ NOTE: currently this function does not padding for post length. """
    _, text_word_size, query_word_size, code_token_size = bucket
    iid, text_blocks, text_block_actual_lengths, code_blocks, code_block_actual_lengths, query, query_length,\
      code_label, code_idx, post_actual_length = instance

    # padded blocks
    new_text_blocks = [padding_inputs(b, text_word_size) for b in text_blocks]
    new_code_blocks = [padding_inputs(b, code_token_size) for b in code_blocks]
    new_query = padding_inputs(query, query_word_size)
    new_query_length = min(query_length, query_word_size)

    # adjust the actual lengths: should shorter than the maximum
    new_text_block_actual_lengths = [min(i, text_word_size) for i in text_block_actual_lengths]
    new_code_block_actual_lengths = [min(i, code_token_size) for i in code_block_actual_lengths]

    new_instance = [iid, new_text_blocks, new_text_block_actual_lengths, new_code_blocks, new_code_block_actual_lengths,
                    new_query, new_query_length, code_label, code_idx, post_actual_length]

    return new_instance

  count_truncated = 0
  count_failed = 0
  for data_i in iid_data:
    text_blocks = data_i[1]
    code_blocks = data_i[3]
    query = data_i[5]

    max_text_len = max([len(b) for b in text_blocks])
    max_code_len = max([len(b) for b in code_blocks])
    query_len = len(query)

    if max_text_len > buckets[-1][1] or query_len > buckets[-1][2] or max_code_len > buckets[-1][3]:
      if truncate_on:
        data_in_buckets[-1].append(padding_data_instance(data_i, buckets[-1]))
        count_truncated += 1
      else:
        count_failed += 1
    else:
      bucket_idx = min([i for i in range(len(buckets)) if max_text_len <= buckets[i][1] and
                        query_len <= buckets[i][2] and max_code_len <= buckets[i][3] and
                        len(text_blocks) <= buckets[i][0]])
      data_in_buckets[bucket_idx].append(padding_data_instance(data_i, buckets[bucket_idx]))

  data_in_buckets_size = [len(_) for _ in data_in_buckets]
  print("Bucket: %s" % str(buckets))
  print("Data size: \n In total: %d.\nBucket size: %s." % (
    sum(data_in_buckets_size), str(data_in_buckets_size)))
  print("%d examples failed." % count_failed)
  print("%d examples get truncated." % count_truncated)

  return data_in_buckets


########################################
#### update 09/08: analyzing predictions
########################################

def eval_basic(prediction_data):
  tp, tn, fp, fn = [], [], [], []
  for iid, (gold, unscaled_probs) in prediction_data:
    if gold == 1:
      if unscaled_probs[1] > unscaled_probs[0]:
        tp.append((iid, (gold, unscaled_probs)))
      else:
        fn.append((iid, (gold, unscaled_probs)))
    else:
      if unscaled_probs[1] > unscaled_probs[0]:
        fp.append((iid, (gold, unscaled_probs)))
      else:
        tn.append((iid, (gold, unscaled_probs)))

  tp_tn_fp_fn_tuple = (tp, tn, fp, fn)

  prec = len(tp) * 1.0 / (len(tp) + len(fp))
  recall = len(tp) * 1.0 / (len(tp) + len(fn))
  f1 = 2 * prec * recall / (prec + recall)
  accuracy = (len(tp) + len(tn)) * 1.0 / (len(tp) + len(tn) + len(fp) + len(fn))

  # print info
  print("Total number of examples: %d." % len(prediction_data))
  print("tp: %d, tn: %d, fp: %d, fn: %d.\nprecision: %.3f, recall: %.3f, f1: %.3f, accuracy: %.3f." % (
    len(tp), len(tn), len(fp), len(fn), prec, recall, f1, accuracy
  ))

  return tp_tn_fp_fn_tuple


def analysis_pipeline(prediction_data, iid2content, qid2title, iid2code_tokens, savedir):
  # get tp, tn, fp, fn tuple first
  tp_tn_fp_fn_tuple = eval_basic(prediction_data)
  pickle_save(tp_tn_fp_fn_tuple, savedir + "tp_tn_fp_fn_tuple.pickle")

  # sample some predictions
  samples = [random.sample(x, min(10, len(x))) for x in tp_tn_fp_fn_tuple]
  for idx, sample in enumerate(samples):
    if idx == 0:
      filename = savedir + "tp.txt"
    elif idx == 1:
      filename = savedir + "tn.txt"
    elif idx == 2:
      filename = savedir + "fp.txt"
    elif idx == 3:
      filename = savedir + "fn.txt"
    else:
      raise ValueError("Invalid statistics size.")

    with open(filename, "w") as f:
      for iid, _ in sample:
        qid, code_idx = iid
        title = qid2title[qid]
        contexts = iid2content[iid][1]
        f.write("Question title: %s\nAnswer code index (starting from 0): %d.\n" % (
          title.encode("ascii", "ignore"), code_idx))
        f.write("Link: ")
        f.write("https://stackoverflow.com/questions/%d\n" % qid)
        f.write("Pre-context: \"%s\"\n" % " ".join([item.encode("ascii", "ignore") for item in contexts[0]]))
        f.write("Post-context: \"%s\"\n" % " ".join([item.encode("ascii", "ignore") for item in contexts[1]]))
        f.write("Code tokens: \"%s\"\n\n" % " ".join([item.encode("ascii", "ignore") for item in iid2code_tokens[iid]]))


def select_first(iid_data):
  tp, tn, fp, fn = 0, 0, 0, 0

  for datai in iid_data:
    qid, idx = datai[0]
    label = datai[7]

    if label == 1:
      if idx == 0: # pred: 1
        tp += 1
      else: # pred: 0
        fn += 1
    else:
      if idx == 0: # pred: 1
        fp += 1
      else: # pred: 0
        tn += 1

  precision = tp*1.0 / (tp + fp)
  recall = tp * 1.0 / (tp + fn)
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = (tp + tn) * 1.0 / (tp + tn + fp + fn)

  print("Precision, recall, f1, accuracy:\n%.3f\t%.3f\t%.3f\t%.3f" % (precision, recall, f1, accuracy))


def select_all(iid_data):
  precision = len([1 for datai in iid_data if datai[7] == 1]) * 1.0 / len(iid_data)
  recall = 1.0
  f1 = 2 * precision * recall / (precision + recall)
  accuracy = precision

  print("Precision, recall, f1, accuracy:\n%.3f\t%.3f\t%.3f\t%.3f" % (precision, recall, f1, accuracy))


def explore_info(iids, qid2title, iid2code_tokens):
  for iid in iids:
    title = qid2title[iid[0]]
    code_tokens = iid2code_tokens[iid]
    print(iid)
    print(title)
    print(code_tokens)
    print("=" * 10)


def show_cloest_items(iids, iid2vec, iid2info):
  vecs = [iid2vec[iid] for iid in iids]
  dist_matrix = squareform(pdist(vecs, 'cosine'))
  sorted_dist_matrix = np.argsort(dist_matrix)

  sampled_indices = random.sample(range(len(iids)), 20)
  for idx in sampled_indices:
    iid = iids[idx]
    # assert sorted_dist_matrix[idx][0] == idx # the first is itself
    sorting_lists = sorted_dist_matrix[idx][1:11]
    print("Target: %s\n%s\nThe most similar items: " % (str(iid), iid2info[iid]))
    for close_idx in sorting_lists:
      print("iid: %s, distance: %.3f\ninfo: %s\n" % (
        iids[close_idx], dist_matrix[idx][close_idx], iid2info[iids[close_idx]]))
    print("=" * 20)


########################################################
##### update 09/18: for model ensemble
########################################################

def combine_vote(pred_list, strategy, bool_eval=True):
  """
  Given a list of predition lists, make a final prediction based on different voting strategies.
  Args:
    pred_list: a list of prediction data from different models.
    strategy: supporting "vote", "multiply", "agree".

  Returns:
    pred_vote: a dict of {iid: prediction}.

  """
  # data pre-processing
  _pred_list = []
  for pred_data in pred_list:
    _pred_data = {item[0]:item[1] for item in pred_data}
    if bool_eval:
      _ = eval_basic(_pred_data.items())
    _pred_list.append(_pred_data)

  iids = _pred_list[0].keys() # all candidates
  num_models = len(pred_list) # number of models
  print("Combining %d models." % num_models)

  pred_vote = dict() # final prediction

  for iid in iids:
    gold = _pred_list[0][iid][0]
    if strategy == "vote":
      label_list = [1 if pred[iid][1][1] > pred[iid][1][0] else 0 for pred in _pred_list]
      if np.average(label_list) > 0.5:
        pred_vote[iid] = (gold, (0, 1))
      else:
        pred_vote[iid] = (gold, (1, 0))
    elif strategy == "multiply":
      probability_list = [softmax(pred[iid][1]) for pred in _pred_list]
      probs = np.prod(np.array(probability_list), axis=0)
      probs = probs / sum(probs)
      pred_vote[iid] = (gold, probs)
    elif strategy == "agree":
      label_list = [1 if pred[iid][1][1] > pred[iid][1][0] else 0 for pred in _pred_list]
      if label_list.count(1) == len(label_list):
        pred_vote[iid] = (gold, (0, 1))
      elif label_list.count(0) == len(label_list):
        pred_vote[iid] = (gold, (1, 0))
    else:
      raise Exception("Invalid argument strategy!")

  print(len(pred_vote))

  if bool_eval:
    tp_tn_fp_fn_tuple = eval_basic(pred_vote.items())
  else:
    tp_tn_fp_fn_tuple = None

  return tp_tn_fp_fn_tuple, pred_vote


def prepare_tri_training_data(pred_list, size, prob_threshold=0.0):
  """
  Return annotations agreed by models in the list.
  Args:
    pred_list: a list of models, each is a list of (iid, (0, pred), bucket_id)
    size: the demanded data size.
    prob_threshold: used to filter examples with low probabilities.

  Returns:
    agreed_data: a bucket-sized dataset, where each instance is (iid, label).

  """
  # preprocess to {iid: (iid, (0, pred), bucket_id)}
  _pred_list = []
  for pred_data in pred_list:
    _pred_data = {item[0]: item for item in pred_data}
    _pred_list.append(_pred_data)

  iids = _pred_list[0].keys()

  agreed_pos_data = [[] for _ in range(4)]
  agreed_neg_data = [[] for _ in range(4)]
  agreed_pos_iids = []
  agreed_neg_iids = []
  pos_size = size // 2
  neg_size = size - pos_size

  # analyzing each instance
  for iid in iids:
    bucket_id = _pred_list[0][iid][-1]
    model_annotations = []
    bool_fail = False
    for model in _pred_list:
      pred = softmax(model[iid][1][1])
      if pred[1] > pred[0]:
        if pred[1] >= prob_threshold:
          model_annotations.append(1)
        else:
          bool_fail = True
          break
      else:
        if pred[0] >= prob_threshold:
          model_annotations.append(0)
        else:
          bool_fail = True
          break
    if not bool_fail: # all confident
      if len(model_annotations) == model_annotations.count(1): # both 1
        agreed_pos_data[bucket_id].append((iid, 1))
        agreed_pos_iids.append(iid)
      elif len(model_annotations) == model_annotations.count(0): # both 0
        agreed_neg_data[bucket_id].append((iid, 0))
        agreed_neg_iids.append(iid)

  if len(agreed_pos_iids) > pos_size:
    print("Down sampling for positive data...")
    sampled_iids = set(random.sample(agreed_pos_iids, pos_size))
    agreed_pos_data = [[item for item in bucket if item[0] in sampled_iids] for bucket in agreed_pos_data]

  if len(agreed_neg_iids) > neg_size:
    print("Down sampling for negative data...")
    sampled_iids = set(random.sample(agreed_neg_iids, neg_size))
    agreed_neg_data = [[item for item in bucket if item[0] in sampled_iids] for bucket in agreed_neg_data]

  agreed_data = [item1 + item2 for item1, item2 in zip(agreed_pos_data, agreed_neg_data)]
  show_bucket_data_stat(agreed_data, 1)

  return agreed_data


def bucket_iid_to_real_data(bucket_data_in_iid, source_bucket_data):
  """
  Collect real data from source_bucket_data, given bucket_data_in_iid.
  Args:
    bucket_data_in_iid: a bucket-sized data, each contains a list of (iid, label).
    source_bucket_data: the real bucket-sized data.

  Returns:
    real_data.

  """
  real_data = []

  for bucket_id, data in enumerate(source_bucket_data):
    # preprocess
    data_dict = {item[0]:item for item in data}
    iid_data = bucket_data_in_iid[bucket_id]

    real_bucket = []
    for iid, label in iid_data:
      data_dict[iid][-3] = label
      real_bucket.append(data_dict[iid])

    real_data.append(real_bucket)

  show_bucket_data_stat(real_data, -3)

  return real_data


###############################################
##### Looking for similar code snippets
###############################################
def collect_n_grams(tokens, gram_level):
  if gram_level == 1:
    return tokens
  else:
    return zip(*[tokens[i:] for i in range(gram_level)])

def compare_two_code_snippets(code_tokens_1, code_tokens_2, gram_level=1):
  """
  Comparing two code snippets.
  Args:
    code_tokens_1: a list of tokens.
    code_tokens_2: a list of tokens.
    gram_level: the <=n-gram level to compare.

  Returns:
    similarity score: #of overlapping terms / #of all terms in the two code snippets. within [0, 1].

  """
  if len(code_tokens_1) == 0 or len(code_tokens_2) == 0:
    return 0.0

  terms_1 = []
  terms_2 = []
  for n in range(1, gram_level+1):
    terms_1.extend(collect_n_grams(code_tokens_1, n))
    terms_2.extend(collect_n_grams(code_tokens_2, n))

  score = len(set(terms_1).intersection(terms_2)) * 1.0 / max(len(set(terms_1)), len(set(terms_2)))
  return score

def looking_for_similar_code(postlink_records, qid2code_tokens, gram_level, threshold):
  """
  Looking for similar code snippets.
  Args:
    postlink_records: a list of (qid, related_qid, id, date, related_type).
    qid2code_tokens: {qid/iid: code tokens}.
    gram_level: set the gram level for measuring the similarity of two code snippets.
    threshold: the threshold to decide similarity.

  Returns:
    a pair of (qid/iid, related qid/iid with similar code). We perform cosine similarity.

  """
  # preprocess
  all_qids = set()
  all_iids_dict = dict() # {qid: a set of indices}
  for id in qid2code_tokens.keys():
    if isinstance(id, tuple):
      all_iids_dict[id[0]] = all_iids_dict.get(id[0], set()).union({id[1]})
    else:
      all_qids.add(id)

  # helper function
  def get_code(id):
    if id in all_qids:
      return [(id, qid2code_tokens[id])]
    elif id in all_iids_dict:
      code_list = []
      for idx in all_iids_dict[id]:
        code_list.append(((id, idx), qid2code_tokens[(id, idx)]))
      return code_list
    else:
      return None

  similar_pairs = [] # results
  # start
  for qid, related_qid, _, _, related_type in postlink_records:
    qid = int(qid)
    related_qid = int(related_qid)
    code_list = get_code(qid)
    related_code_list = get_code(related_qid)
    if code_list is None or related_code_list is None:
      continue

    # comparison
    for iid, code in code_list:
      for related_iid, related_code in related_code_list:
        simi_score = compare_two_code_snippets(code, related_code, gram_level)
        if simi_score > threshold:
          similar_pairs.append((iid, related_iid, simi_score))

  print("Found %d pairs!" % len(similar_pairs))
  return similar_pairs


def main():

  lang = "python"

  # update 07/16
  def test_get_vocab_and_text2id():
    savedir = "../annotation_tool/data/code_solution_labeled_data/source/"
    pickle_source = savedir + "%s_how_to_do_it_by_classifier_all_multiple_answer_codes_" \
                              "qid_partialtext_code_label_tuples.pickle" % lang
    pickle_vocab = savedir + "%s_text_content/text_word_vocab.pickle" % lang

    # valid_iids = pickle_load("../data/data_hnn/sql/train/iids.pickle")
    valid_iids = None

    sorted_word_count, vocab, _ = get_vocab(pickle_source, pickle_vocab, min_threshold=2, valid_iids=valid_iids)

    pickle_save(sorted_word_count, savedir + "%s_text_content/sorted_word_count.pickle" % lang)

    _ = data_to_symbols(pickle_source, pickle_vocab,
                        savedir + "%s_text_content/qid_texts2ids_codes_labels.pickle" % lang)

  def test_get_new_hnn_format_single_code():
    source_dir = "../annotation_tool/data/code_solution_labeled_data/source/"

    dir = "../data/data_hnn/%s/disagreed/" % lang
    # iid_to_label = pickle_load("../annotation_tool/crowd_sourcing/%s_annotator/all_agreed_iid_to_label.pickle" % lang)
    all_iids = pickle_load(dir + "iids.pickle")

    all_qid_to_idx = dict()
    for iid in all_iids:
      qid, idx = iid
      if qid in all_qid_to_idx:
        all_qid_to_idx[qid].append(idx)
      else:
        all_qid_to_idx[qid] = [idx]

    qid_texts2ids_codes_labels = pickle_load(source_dir + "%s_text_content/qid_texts2ids_codes_labels.pickle" % lang)
    qid_to_code = pickle_load(source_dir + "%s_code_gram5/qid_to_tokenized_code_id.pickle" % lang)
    qid_to_query = pickle_load(source_dir + "%s_query_gram5/qid_to_tokenized_query_id_shared_text_vocab.pickle" % lang)

    iid_and_content = get_new_hnn_format_single_code(
      qid_texts2ids_codes_labels, qid_to_code, qid_to_query, all_qid_to_idx=all_qid_to_idx, iid_to_label=None)
    print("Saving...")
    pickle_save(iid_and_content, dir + "iid_data_partialcontext_shared_text_vocab.pickle")

  def test_get_new_hnn_format_bucket():
    buckets = [(2, 10, 22, 72), (2, 20, 34, 102), (2, 40, 34, 202), (2, 100, 34, 302)]
    dir = "../data/data_hnn/%s/disagreed/" % lang
    iid_data = pickle_load(dir + "iid_data_partialcontext_shared_text_vocab.pickle")
    # iid_data = pickle_load(dir + "iid_data_partialcontext_python_shared_text_code_vocab.pickle")
    data_in_buckets = get_new_hnn_format_bucket(iid_data, buckets, truncate_on=True)

    print("Saving...")
    pickle_save(data_in_buckets, dir + "data_partialcontext_shared_text_vocab_in_buckets.pickle")

  def test_get_pretrained_word_vec():
    savedir = "../annotation_tool/data/code_solution_labeled_data/source/sql_query_gram5/"
    vocab = pickle_load(savedir + "query_word_vocab.pickle")

    # word embedding
    glove_dict = pickle.load(open("../data/sql_glove_embedding_min2_300.pickle", "rb"))

    emb = get_pretrained_word_vec(glove_dict, vocab, 300)
    pickle_save(emb, savedir + "rnn_word_embedding_300.pickle")

  def test_get_HAN_data():
    pickleannotation = "../annotation_tool/data/source_data/sql_missed_qids.pickle"

    qid_text_code_codelabel_tuples, qid_partialtext_code_codelabel_tuples =\
      get_HAN_data("../annotation_tool/data/source_data/"
        "sql_question_id_to_title_and_answer_text_code_blocks_and_question_text_code_blocks.dict", # for single
        pickleannotation)
    with open("../annotation_tool/data/code_solution_labeled_data/source/"
      "sql_how_to_do_it_by_classifier_all_multiple_answer_codes_qid_text_code_label_tuples_missed.pickle",
      "wb") as f:
      pickle.dump(qid_text_code_codelabel_tuples, f)
    with open("../annotation_tool/data/code_solution_labeled_data/source/"
      "sql_how_to_do_it_by_classifier_all_multiple_answer_codes_qid_partialtext_code_label_tuples_missed.pickle",
      "wb") as f:
      pickle.dump(qid_partialtext_code_codelabel_tuples, f)

  def test_analysis_pipeline():
    dir = "../data/data_hnn/%s/checkpoint/" % lang
    prediction_data = pickle_load(dir + "/ckpt_data_partialcontext_shared_text_vocab_RNN_block128_input128_code_mlp1_units128_keepprob0.700_lr0.001_bs100_seed1_l2reg0.000_text1_setting64-150-24379-0-1-0-1_code1_setting64-150-218900-0-1-0-1_query3_setting64-150-24379-0-1-0-1/validation_result.pickle")
    source_data = pickle_load("../data/data_hnn/%s/dev_iid_partialcontexts_code_label.pickle" % lang)
    iid2content = {datai[0]: datai for datai in source_data}

    # _source_data = pickle_load("../data/data_hnn/%s/train/adverserial_training_precontext_replaced.pickle" % lang)
    # vocab = pickle_load("../annotation_tool/data/code_solution_labeled_data/source/%s_text_content/text_word_vocab.pickle" % lang)
    # iid2content = dict()
    # for item in _source_data:
    #   iid = item[0]
    #   context_ids = item[1]
    #   contexts = symbols_to_data(vocab, context_ids)
    #   code_ids = item[3]
    #   label = item[-3]
    #   iid2content[iid] = (iid, contexts, code_ids, label)

    iid2code_tokens = pickle_load("../data/data_hnn/%s/dev_iid2code_tokens.pickle" % lang)
    qid2title = pickle_load("../annotation_tool/data/code_solution_labeled_data/source/"
                            "%s_how_to_do_it_by_classifier_multiple_qid_to_title.pickle" % lang)

    # savedir = "../data/data_hnn/%s/analysis/training_adv_precontext_111/" % lang
    savedir = "../data/data_hnn/%s/analysis/validation_111/" % lang
    analysis_pipeline(prediction_data, iid2content, qid2title, iid2code_tokens, savedir)

  def test_baselines():
    iid_data = pickle_load("../data/data_hnn/sql/test/iid_data_partialcontext.pickle")
    select_first(iid_data)
    select_all(iid_data)

  def test_explore_info():
    iid2code_tokens = pickle_load("../data/data_hnn/sql/dev_iid2code_tokens.pickle")
    qid2title = pickle_load("../annotation_tool/data/code_solution_labeled_data/source/"
                            "sql_how_to_do_it_by_classifier_multiple_qid_to_title.pickle")
    iid2label = pickle_load("../annotation_tool/crowd_sourcing/sql_annotator/all_agreed_iid_to_label.pickle")
    iids = pickle_load("../data/data_hnn/sql/valid/iids.pickle")
    iids = [iid for iid in iids if iid2label[iid] == 1]
    explore_info(random.sample(iids, 20), qid2title, iid2code_tokens)

  def test_show_cloest_items():
    iid2query_sem, iid2code_sem, iid2ff_output = pickle_load("../data/data_hnn/%s/analysis/validation_data_022/"
                               "iid2query_sem_iid2code_sem_iid2ff_output.pickle" % lang)
    iids = iid2code_sem.keys()

    # qid2title = pickle_load("../annotation_tool/data/code_solution_labeled_data/source/"
    #                         "%s_how_to_do_it_by_classifier_multiple_qid_to_title.pickle" % lang)
    iid2code_tokens = pickle_load("../data/data_hnn/%s/dev_iid2code_tokens.pickle" % lang)
    show_cloest_items(iids, iid2query_sem, iid2code_tokens)

  def test_eval_basic():
    pred_011 = pickle_load("../data/data_hnn/python/checkpoint/"
                           "ckpt_data_partialcontext_shared_text_vocab__code_mlp1_units128_keepprob1.000_lr0.001_bs100_seed1_l2reg0.000_text0_settingNone_code1_setting64-150-218900-0-1-0-1_query1_setting64-150-24379-0-1-0-1/validation_result.pickle")
    pred_data = [item[:2] for item in pred_011]
    _ = eval_basic(pred_data)

  def test_combine_vote():
    item = "disagreed" # or "test", "validation"

    # sql
    dir = "../data/data_hnn/sql/checkpoint/"

    # initial
    pred_100 = pickle_load("../data/data_hnn/sql/checkpoint_tanh/ckpt_data_partialcontext_block_size_128_bidirection_1_variant_1_keep_prob_0.700_lr_0.001_bs_100_seed_1_l2reg_0.000_text_model_1_setting_64-150-13698-0-1-0-1_code_model_0_setting_None_query_model_0_setting_None/%s_result.pickle" % item)
    pred_011 = pickle_load(dir + "ckpt_data_partialcontext_shared_text_vocab__code_mlp1_units128_keepprob0.500_lr0.001_bs100_seed1_l2reg0.000_text0_settingNone_code1_setting64-150-33192-0-1-0-1_query1_setting64-150-13698-0-1-0-1/%s_result.pickle" % item)
    pred_111 = pickle_load(dir + "ckpt_data_partialcontext_shared_text_vocab_RNN_block128_input128_code_mlp1_units128_keepprob0.700_lr0.001_bs100_seed1_l2reg0.000_text1_setting64-150-13698-0-1-0-1_code1_setting64-150-33192-0-1-0-1_query3_setting64-150-13698-0-1-0-1/%s_result.pickle" % item)

    # # python
    # dir = "../data/data_hnn/python/checkpoint/"
    #
    # # initial
    # pred_100 = pickle_load(dir +
    #                        "ckpt_data_partialcontext_RNN_block128_input128_keepprob0.500_lr0.001_bs100_seed1_l2reg0.000_text1_setting64-150-24379-0-1-0-1_code0_settingNone_query0_settingNone/%s_result.pickle" % item)
    # pred_011 = pickle_load(dir +
    #                        "ckpt_data_partialcontext_shared_text_vocab__code_mlp1_units128_keepprob1.000_lr0.001_bs100_seed1_l2reg0.000_text0_settingNone_code1_setting64-150-218900-0-1-0-1_query1_setting64-150-24379-0-1-0-1/%s_result.pickle" % item)
    # pred_111 = pickle_load(dir +
    #                        "ckpt_data_partialcontext_shared_text_vocab_RNN_block128_input128_code_mlp1_units128_keepprob0.700_lr0.001_bs100_seed1_l2reg0.000_text1_setting64-150-24379-0-1-0-1_code1_setting64-150-218900-0-1-0-1_query3_setting64-150-24379-0-1-0-1/%s_result.pickle" % item)

    pred_list = [pred_100, pred_011, pred_111]
    agreed_performance, agreed_predictions = combine_vote(pred_list, "agree", bool_eval=False)
    pickle_save(agreed_predictions, "../data/data_hnn/sql/disagreed/agreed_predictions_100_011_111.pickle")

    # analyze_error_overlapping(pred_list)

  def test_looking_for_similar_code():
    postlink_records = pickle_load("../../records_from_PostLinks.pickle")
    qid2code_tokens = pickle_load("../annotation_tool/data/code_solution_labeled_data/source/"
                                  "sql_code_gram5/qid_to_tokenized_code.pickle")
    threshold = 0.8
    gram_level = 2
    code_pairs = looking_for_similar_code(postlink_records, qid2code_tokens, gram_level, threshold)
    pickle_save(code_pairs, "../data/data_hnn/sql/tri-training/similar_code_pairs_%.3f.pickle" % threshold)


  # test_get_vocab_and_text2id()
  # test_get_new_hnn_format_single_code()
  # test_get_new_hnn_format_bucket()
  # test_get_pretrained_word_vec()
  # test_get_HAN_data()
  # test_analysis_pipeline()
  # test_baselines()
  # test_explore_info()
  # test_show_cloest_items()
  test_combine_vote()
  # test_looking_for_similar_code()


if __name__ == "__main__":
  main()




