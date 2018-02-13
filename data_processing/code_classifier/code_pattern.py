
import re
from bs4 import BeautifulSoup as bs
import pickle
import datetime
import numpy as np

import pdb


## python/c++/java operators
## collected from tutorialpoint.com
ARITH_OP = {'+', '-', '*', '/', '%', '**', '//', '++', '--'}
COMPARE_OP = {'==', '!=', '<>', '<', '>', '<=', '>='}
ASSIGN_OP = {'=', '+=', '-=', '*=', '/=', '%=', '**=', '//=', '>>=', '<<=', '&=', '|=', '^='}
BITWISE_OP = {'&', '|', '^', '~', '<<', '>>', '>>>'}
LOGICAL_OP = {' and ', ' or ', ' not ', '&&', '||', '!'}
MEMBER_OP = {' in ', ' not in '}
IDENTITY_OP = {' is ', ' is not '}
MISC_OP = {'?:', ':', '?', '&', '*', '->', '::'}
# other_op = { ',', '.', '\"', '\''}

SET_OPERATORS = set().union(ARITH_OP).union(COMPARE_OP).union(ASSIGN_OP).union(BITWISE_OP).union(LOGICAL_OP).\
  union(MEMBER_OP).union(IDENTITY_OP).union(MISC_OP)

# TODO: currently: python. if not enough, we may consider other programming languages
# PATTERN_DEFINE_FUNC = re.compile("def .*\(.*\):") #cannot deal with multi-line definition
PATTERN_DEFINE_FUNC = re.compile("def \S+\([^\(\)]*\)", re.DOTALL)
PATTERN_DEFINE_CLASS = re.compile("class \S+\([^\(\)]*\)", re.DOTALL)
#PATTERN_CALL_FUNC = re.compile("(\..+\(.*\))|(\..+\[.+\])") # didn't define the func name, cannot handle the line separation, cannot handle function call without ()
PATTERN_CALL_FUNC = re.compile("(\S+\(.*\))|(\S+\[\S+\])|(\D\.\D)", re.DOTALL)
PATTERN_TRY_EXCEPT = re.compile("try:.+(error|exception)", re.DOTALL)
PATTERN_ERROR_MESSAGE = re.compile("(error|exception)[^a-z]")
# PATTERN_DEFINE_FUNC_EXCEPT = re.compile("def (\s|.)+(error|exception)")
# PATTERN_CALL_FUNC_EXCEPT = re.compile("[a-z]\([^\)]*error[^\(]*\)")

PATTERN_OTHER_CALL = re.compile("((python)|(import)|(pip)|(print))\s+\S+")
PATTERN_SQL_COMMAND = re.compile("^((SELECT)|(CREATE)|(DELETE)|(UPDATE)|(INSERT)|(ALTER)|(DROP))", re.IGNORECASE)


def match_case_three(code):
  """ Given a code snippet, see whether it is an error message.
  Args:
    code: a string. The code snippet.
  Returns:
    boolMatch: return True if it matches the error message pattern. """

  if PATTERN_CALL_FUNC.search(code) or PATTERN_DEFINE_FUNC.search(code) or\
    PATTERN_DEFINE_CLASS.search(code) or PATTERN_SQL_COMMAND.search(code) or\
    PATTERN_OTHER_CALL.search(code): # calling a function
      return False

  boolMatch = False
  if PATTERN_ERROR_MESSAGE.search(code.lower()) and not PATTERN_TRY_EXCEPT.search(code.lower()):
    boolMatch = True

  return boolMatch