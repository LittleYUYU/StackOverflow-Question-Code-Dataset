# features.py 
# Collecting features for question type classifier.


import pickle
from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize
import re
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import sys

from code_classifier import lr, code_pattern


SQL_KEYWORDS = {"select", "where", "and", "or", "not", "order by", "insert into", "update", "delect",
                "count", "avg", "sum", "max", "min", "like", "between", "in", "join", "inner join", "left join",
                "right join", "full join", "self join", "union", "group by", "having", "exists", "create",
                "drop", "alter", "unique", "primary key", "foreign key", "check", "default", "index",
                "not null", "view"}


# load keywords
with open("howto_kw.txt", "r") as f:
  howto_kw = [term.strip() for term in f.readlines()]
howto_kw_weight_pos = 7
with open("debug_kw.txt", "r") as f:
  debug_kw = [term.strip() for term in f.readlines()]
debug_kw_weight_pos = 14
with open("other_kw.txt", "r") as f:
  other_kw = [term.strip() for term in f.readlines()]
other_kw_weight_pos = 18


def code_tokenize(code):
  return [_.strip() for _ in re.split(r"([.:,\n\t( )=\+\-\*\/])", code) if _.strip()]

def helper_k_gram(list_of_tokens, k):
  k_grams = []
  for i in range(len(list_of_tokens)-k+1):
    k_grams.append(" ".join(list_of_tokens[i:i+k]))
  return k_grams

def simi_ans_to_question(code, list_of_reference_codes, k):

  code_tokens = set(helper_k_gram(code_tokenize(code),k))
  
  if len(code_tokens) == 0:
    return 0.0

  list_of_reference_code_tokens = [set(helper_k_gram(code_tokenize(code), k))
    for code in list_of_reference_codes]

  overlaps = [len(code_tokens.intersection(reference_code_tokens))
    for reference_code_tokens in list_of_reference_code_tokens]
  simi_score = sum(overlaps)*1.0/len(code_tokens)
  return simi_score

def keyword_featuring(text, keyword_list, weighting=False, weighting_type=None):
  text_tokens = text.split()
  feature = dict()
  for idx in range(len(keyword_list)):
    term = keyword_list[idx]
    keywords = term.split("\t")
    count = 0
    for keyword in keywords:
      if ".*" in keyword:
        count += len(re.findall(keyword.decode('utf-8'), text))
      else:
        count += text_tokens.count(keyword.decode('utf-8'))
    if weighting:
      if weighting_type == "howto" and idx < howto_kw_weight_pos:
        count = 5*count
      if weighting_type == "debug" and idx < debug_kw_weight_pos:
        count = 5*count
      if weighting_type == "other" and idx < other_kw_weight_pos:
        count = 5*count
    feature[term] = count
  return feature

def working_code_classifier(list_of_codes, model):
  code_dict = dict()
  for i,code in enumerate(list_of_codes):
    code_dict[i] = code
  features = lr.featuring(code_dict, False, "code_classifier/code_classifier_training_data_featuremap.dict")
  
  pred_dict = dict() #predict result
  for i, feature in features.items():
    pred_dict[i] = float(model.predict_proba(np.array(feature).reshape(1, -1))[:,1][0])

  return pred_dict

def heuristic_working_code_identifier(list_of_codes, keyword_set, threshold=1):
  list_bool_working_code = []
  for i,code in enumerate(list_of_codes):
    code_tokens = [token.lower() for token in code_tokenize(code)]
    num_inter = sum([1 for token in code_tokens if token in keyword_set])
    if num_inter >= threshold:
      list_bool_working_code.append(1)
    else:
      list_bool_working_code.append(0)
  return list_bool_working_code


def featuring(instance, code_type):
  """ Feature engineering.
  Args:
    instance: {title, text_code_blocks, question_post_text_code_blocks} in dict format,
      where the text_code_blocks is a list of text and code blocks (in string) extracted from the answer post,
      and the question_post_text_code_blocks is a list of text and code blocks (in string) extracted from the question post.
    code_type: either python or sql.
  Returns:
    feature: a dict of {feature: value}. """
  assert code_type in {"python", "sql"}

  feature = dict()

  if code_type == "python":
    # load working_code_classifier
    model = pickle.load(open("code_classifier/code_classifier.model"))


  ans_text_code_blocks = instance['answer_text_code_blocks']
  question_text_code_blocks = instance['question_text_code_blocks']

  boolAHasInlineCode, boolQHasInlineCode = 0, 0

  # ans text/code
  ans_text, ans_code = [], []
  for term in ans_text_code_blocks:
    soup = bs(term, "html.parser")
    if "</code></pre>" in term:
      # ans_code.append(re.sub("<.*pre><code>", "", term.replace("</code></pre>", "")).decode('utf-8'))
      try:
        ans_code.append(soup.text.decode('utf-8', 'ignore'))
      except UnicodeEncodeError:
        ans_code.append(soup.text)
    else:
      # ans_text.append(term.replace("<p>", "").replace("</p>", "").replace("<ol>", "").replace("</ol>", "").replace("<ul>", "").replace("</ul>", ""))
      try:
        ans_text.append(soup.text.decode('utf-8','ignore'))
      except UnicodeEncodeError:
      	ans_text.append(soup.text)
      # detect inline code in answers
      if re.search("<code>.*</code>", term):
      	boolAHasInlineCode = 1
  ans_text_tokenized = [word_tokenize(text.lower()) for text in ans_text]


  # question text/code
  question_text, question_code = [], []
  for term in question_text_code_blocks:
    soup = bs(term, "html.parser")
    if "</code></pre>" in term:
      # question_code.append(re.sub("<.*pre><code>", "", term.replace("</code></pre>", "")).decode('utf-8'))
      try:
        question_code.append(soup.text.decode('utf-8', 'ignore'))
      except UnicodeEncodeError:
      	question_code.append(soup.text)
    else:
      # question_text.append(term.replace("<p>", "").replace("</p>", "").replace("<ol>", "").replace("</ol>", "").replace("<ul>", "").replace("</ul>", ""))
      try:
        question_text.append(soup.text.decode('utf-8', 'ignore'))
      except UnicodeEncodeError:
      	question_text.append(soup.text)
      if re.search("<code>.*</code>", term):
      	boolQHasInlineCode = 1
  question_text_tokenized = [word_tokenize(text.lower()) for text in question_text]

  
  # keyword feature
  keyword_feature = keyword_featuring(
    " ".join([" ".join(text) for text in ans_text_tokenized] + [" ".join(text) for text in question_text_tokenized]),
    howto_kw + debug_kw + other_kw)
  feature = keyword_feature

  feature['boolQHasInlineCode'] = boolQHasInlineCode
  feature['boolAHasInlineCode'] = boolAHasInlineCode

  howto_kw_feature = sum(list(keyword_featuring(
    " ".join([" ".join(text) for text in ans_text_tokenized] + [" ".join(text) for text in question_text_tokenized]),
    howto_kw, True, "howto").values()))
  feature['howto_kw_feature'] = howto_kw_feature
  debug_kw_feature = sum(list(keyword_featuring(
    " ".join([" ".join(text) for text in ans_text_tokenized] + [" ".join(text) for text in question_text_tokenized]),
    debug_kw, True, "debug").values()))
  feature['debug_kw_feature'] = debug_kw_feature
  other_kw_feature = sum(list(keyword_featuring(
    " ".join([" ".join(text) for text in ans_text_tokenized] + [" ".join(text) for text in question_text_tokenized]),
    other_kw, True, "other").values()))
  feature['other_kw_feature'] = other_kw_feature
  
  try:
    title = instance['title'].decode('utf-8', 'ignore')
  except UnicodeError:
    title = instance['title']

  howto_title_kw_feature = sum(list(keyword_featuring(title, howto_kw, True, "howto").values()))
  feature['howto_title_kw_feature'] = howto_title_kw_feature
  debug_title_kw_feature = sum(list(keyword_featuring(title, debug_kw, True, "debug").values()))
  feature['debug_title_kw_feature'] = debug_title_kw_feature
  other_title_kw_feature = sum(list(keyword_featuring(title, other_kw, True, "other").values()))
  feature['other_title_kw_feature'] = other_title_kw_feature

  # meta data
  len_q_text = 0
  for text in question_text_tokenized:
    len_q_text += len(text)
  feature['len_q_text'] = len_q_text
  
  num_question_mark = sum([text_tokens.count("?") for text_tokens in question_text_tokenized])
  feature['num_question_mark'] = num_question_mark

  len_a_text = 0
  for text in ans_text_tokenized:
    len_a_text += len(text)
  feature['len_a_text'] = len_a_text

  num_q_code = len(question_code)
  feature['num_q_code'] = num_q_code
  num_a_code = len(ans_code)
  feature['num_a_code'] = num_a_code
  if len(question_code) == 0:
    max_q_code_length = 0
  else:
    max_q_code_length = max([len(code) for code in question_code])
  feature['max_q_code_length'] = max_q_code_length

  if len(ans_code) == 0: 
    return None
  else:
    max_a_code_length = max([len(code) for code in ans_code])
  feature['max_a_code_length'] = max_a_code_length

  # code feature
  if code_type == "python":
    if len(question_code) == 0:
      feature['num_q_io_code'] = 0
      feature['num_error_code'] = 0
    else:
      q_io_code_pred = working_code_classifier(question_code, model)
      num_q_io_code = len([value for key,value in q_io_code_pred.items() if value < 0.5])
      feature['num_q_io_code'] = num_q_io_code
    
      num_error_code = len([1 for code in question_code if code_pattern.match_case_three(code)])
      feature['num_error_code'] = num_error_code

  elif code_type.lower() == "sql":
    # the code feature for SQL will be number of code snippets containing keywords (sort of working code)
    list_bool_working_question_code = heuristic_working_code_identifier(question_code, SQL_KEYWORDS)
    feature['num_q_io_code'] = list_bool_working_question_code.count(0)
  

  # simi between answer codes and question codes
  if len(question_code) == 0:
    simi_score_unigram = 0.0
    simi_score_bigram = 0.0
  else:
    simi_score_unigram = max([simi_ans_to_question(code, question_code, k=1) for code in ans_code])
    simi_score_bigram = max([simi_ans_to_question(code, question_code, k=2) for code in ans_code])
  feature['simi_score_bigram'] = simi_score_bigram
  feature['simi_score_unigram'] = simi_score_unigram


  return feature


def file_featuring(filepath, savepath, keys=None):
  """ Do featuring work for a file, which contains a dict of {qid, dict of properties}. """
  with open(filepath, "rb") as f:
    data = pickle.load(f)

  featured_data = dict()

  count = 0

  if keys is None:
    keys = list(data.keys())
  
  for qid in keys:
    properties = data[qid]
    count += 1
    if count%1000 == 0:
      print(count)
      sys.stdout.flush()

    feature = featuring(properties)
    if feature is not None:
      featured_data[qid] = feature

  # save
  with open(savepath, "wb") as f:
    pickle.dump(featured_data, f)


