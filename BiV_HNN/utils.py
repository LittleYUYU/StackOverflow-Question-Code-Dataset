# utils.py

import csv
from os import listdir
from os.path import isfile, join
import operator


def mergeDict(dict1, dict2):
  """ Merge two dictionaries by adding values for the same key. """
  for key,value in dict2.items():
    dict1[key] = dict1.get(key, 0) + value
    # if key in dict1.keys():
    #   dict1[key] += value
    # else:
    #   dict1[key] = value

def addTermToDict(key, dict1):
  dict1[key] = dict1.get(key, 0) + 1
  # if key in dict1.keys():
  #   dict1[key] += 1
  # else:
  #   dict1[key] = 1
  return dict1

def loadDataFromFile(filename, boolString=False):
  data = []
  with open(filename, 'r') as f:
    print("Loading data...")
    line = f.readline()
    while line:
      if boolString:
        data.append(line.strip())
      else:
        data.append(eval(line))
      line = f.readline()
    print("In total %d sentences" % len(data))
  return data

def loadCSVDataFromFile(filename):
  with open(filename, "r") as f:
    reader = csv.reader(f)
    data = list(reader)
  return data

def sortDictByValue(x, boolDescend = True):
  """ Sort a dictionary by its value. Return a list of tuples (key, value). """
  sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=boolDescend)  
  return sorted_x

def getFilesInDir(dirname):
  """ Return a list of filenames in the given directory. """
  files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
  return files




