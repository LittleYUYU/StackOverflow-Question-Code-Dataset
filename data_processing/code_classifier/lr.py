import pickle
import re


def featuring(dict_of_codes, isTrain, pickle_featuremap):
  """ Returning data features and its mapping. 
  The features are:
  a) all split tokens.
  b) the proportion of number tokens.
  c) the proportion of tokens containing "(" or ")".
  d) the proportion of tokens containing ".".
  e) the proportion of tokens containing other operators.

  Args:
    dict_of_codes: a dict of {idx: code snippet}.
    isTrain: Set to True for builing the feature map.
    pickle_featuremap: Set the file path to save or load the feature map.
  Returns:
    data_features:
      a dict of {idx: a vector of features}.
    feature_map: a list of features in order."""
  if isTrain:
    feature_map = []
    for idx, code in dict_of_codes.items():
      feature_map.extend(code.split())
    feature_map = list(set(feature_map))
    feature_map.append("PROPORTION_NUM")
    feature_map.append("PROPORTION_PAREN")
    feature_map.append("PROPORTION_DOT")
    feature_map.append("PROPORTION_OPER")
    with open(pickle_featuremap, "wb") as f:
      pickle.dump(feature_map, f, protocol=2)
  else:
    with open(pickle_featuremap, "rb") as f:
      feature_map = pickle.load(f)

  # print("feature map size: %d" % len(feature_map))

  data_features = dict()
  for idx, code in dict_of_codes.items():
    features = [float(0)] * len(feature_map)
    # all tokens
    tokens = code.split()
    length = len(tokens)
    
    number = 0
    paren = 0
    dot = 0
    oper = 0
    for t in tokens:
      if t in feature_map:
        features[feature_map.index(t)] = tokens.count(t)
      if re.match(re.compile("[0-9]+\Z"), t):
        number += 1
      if "(" in t or ")" in t:
        paren += 1
      if "." in t:
        dot += 1
      
      for op in SET_OPERATORS:
        if op in t:
          oper += 1
          break

    if length == 0:
      features[feature_map.index("PROPORTION_NUM")] = 0.0
      features[feature_map.index("PROPORTION_PAREN")] = 0.0
      features[feature_map.index("PROPORTION_DOT")] = 0.0
      features[feature_map.index("PROPORTION_OPER")] = 0.0
    else:
      features[feature_map.index("PROPORTION_NUM")] = float(number)/length
      features[feature_map.index("PROPORTION_PAREN")] = float(paren)/length
      features[feature_map.index("PROPORTION_DOT")] = float(dot)/length
      features[feature_map.index("PROPORTION_OPER")] = float(oper)/length
    
    data_features[idx] = features

  return data_features