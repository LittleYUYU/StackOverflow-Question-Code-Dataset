# data_utils.py
# Processing source data and Preparing the reader.

import token, tokenize
import ast
import re
import sys
import numpy as np
from StringIO import *
from nltk.tokenize import wordpunct_tokenize

import pdb

sys.path.append("codenn/src")
from sql.SqlTemplate import *


PATTERN_VAR_EQUAL = re.compile("(\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)(,\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)*=")
PATTERN_VAR_FOR = re.compile("for\s+[_a-zA-Z][_a-zA-Z0-9]*\s*(,\s*[_a-zA-Z][_a-zA-Z0-9]*)*\s+in")


def repair_program_io(code):
  """ Removing the special IO signs from the program.
  Case1: 
    In [n]:
    (   ....:) 
    and 
    Out [n]: 
  Case2:
    >>> 
    ... 

  Args:
    code: a string, the code snippet.
  Returns:
    repaired_code: a string, the repaired code snippet.
    code_list: a list of strings, each of which is lines of the original code snippet.
      The goal is to maintain all of the original information."""
  
  # reg patterns for case 1
  pattern_case1_in = re.compile("In ?\[\d+\]: ?") # flag1
  pattern_case1_out = re.compile("Out ?\[\d+\]: ?") # flag2
  pattern_case1_cont = re.compile("( )+\.+: ?") # flag3

  # reg patterns for case 2
  pattern_case2_in = re.compile(">>> ?") # flag4
  pattern_case2_cont = re.compile("\.\.\. ?") # flag5

  patterns = [pattern_case1_in, pattern_case1_out, pattern_case1_cont,
    pattern_case2_in, pattern_case2_cont]

  lines = code.split("\n")
  lines_flags = [0 for _ in range(len(lines))] 

  code_list = [] # a list of strings

  # match patterns
  for line_idx in range(len(lines)):
    line = lines[line_idx]
    for pattern_idx in range(len(patterns)):
      if re.match(patterns[pattern_idx], line):
        lines_flags[line_idx] = pattern_idx + 1
        break
  lines_flags_string = "".join(map(str, lines_flags))

  bool_repaired = False

  # pdb.set_trace()
  # repair
  if lines_flags.count(0) == len(lines_flags): # no need to repair
    repaired_code = code
    code_list = [code]
    bool_repaired = True

  elif re.match(re.compile("(0*1+3*2*0*)+"), lines_flags_string) or\
    re.match(re.compile("(0*4+5*0*)+"), lines_flags_string):
    repaired_code = ""
    pre_idx = 0
    sub_block = ""
    if lines_flags[0] == 0:
      flag = 0
      while(flag == 0):
        repaired_code += lines[pre_idx] + "\n"
        pre_idx += 1
        flag = lines_flags[pre_idx]
      sub_block = repaired_code
      code_list.append(sub_block.strip())
      sub_block = "" # clean
    
    for idx in range(pre_idx, len(lines_flags)):
      if lines_flags[idx] != 0:
        repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

        # clean sub_block record
        if len(sub_block.strip()) and (idx > 0 and lines_flags[idx-1] == 0):
          code_list.append(sub_block.strip())
          sub_block = ""
        sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"
      
      else:
        if len(sub_block.strip()) and (idx > 0 and lines_flags[idx-1] != 0):
          code_list.append(sub_block.strip())
          sub_block = ""
        sub_block += lines[idx] + "\n"

    # avoid missing the last unit
    if len(sub_block.strip()):
      code_list.append(sub_block.strip())

    if len(repaired_code.strip()) != 0:
      bool_repaired = True
    
  if not bool_repaired: # not typical, then remove only the 0-flag lines after each Out.
    repaired_code = ""
    sub_block = ""
    bool_after_Out = False
    for idx in range(len(lines_flags)):
      if lines_flags[idx] != 0:
        if lines_flags[idx] == 2:
          bool_after_Out = True
        else:
          bool_after_Out = False
        repaired_code += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

        if len(sub_block.strip()) and (idx > 0 and lines_flags[idx-1] == 0):
          code_list.append(sub_block.strip())
          sub_block = ""
        sub_block += re.sub(patterns[lines_flags[idx] - 1], "", lines[idx]) + "\n"

      else:
        if not bool_after_Out:
          repaired_code += lines[idx] + "\n"

        if len(sub_block.strip()) and (idx > 0 and lines_flags[idx-1] != 0):
          code_list.append(sub_block.strip())
          sub_block = ""
        sub_block += lines[idx] + "\n"


  return repaired_code, code_list


def get_vars(ast_root):
  return sorted({node.id for node in ast.walk(ast_root) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)})


def get_vars_heuristics(code):
  varnames = set()
  code_lines = [_ for _ in code.split("\n") if len(_.strip())]

  # best effort parsing
  start = 0
  end = len(code_lines) - 1
  bool_success = False
  while(not bool_success):
    try:
      root = ast.parse("\n".join(code_lines[start:end]))
    except:
      end -= 1
    else:
      bool_success = True
  # print("Best effort parse at: start = %d and end = %d." % (start, end))
  varnames = varnames.union(set(get_vars(root)))
  # print("Var names from base effort parsing: %s." % str(varnames))

  # processing the remaining...
  for line in code_lines[end:]:
    line = line.strip()
    try:
      root = ast.parse(line)
    except:
      # matching PATTERN_VAR_EQUAL
      pattern_var_equal_matched = re.match(PATTERN_VAR_EQUAL, line)
      if pattern_var_equal_matched:
        match = pattern_var_equal_matched.group()[:-1] # remove "="
        varnames = varnames.union(set([_.strip() for _ in match.split(",")]))

      # matching PATTERN_VAR_FOR
      pattern_var_for_matched = re.search(PATTERN_VAR_FOR, line)
      if pattern_var_for_matched:
        match = pattern_var_for_matched.group()[3:-2] # remove "for" and "in"
        varnames = varnames.union(set([_.strip() for _ in match.split(",")]))

    else:
      varnames = varnames.union(get_vars(root))

  # print("varnames: %s" % str(varnames))

  return varnames


def tokenize_python_code(code):
  bool_failed_var = False
  bool_failed_token = False

  try:
    root = ast.parse(code)
    varnames = set(get_vars(root))
  except:
    repaired_code, _ = repair_program_io(code)
    try:
      root = ast.parse(repaired_code)
      varnames = set(get_vars(root))
    except:
      # failed_var_qids.add(qid)
      bool_failed_var = True
      varnames = get_vars_heuristics(code)

  tokenized_code = []

  def first_trial(_code):
    if len(_code) == 0:
      return True
    try:
      g = tokenize.generate_tokens(StringIO(_code).readline)
      term = g.next()
    except:
      return False
    else:
      return True

  bool_first_success = first_trial(code)
  while not bool_first_success:
    code = code[1:]
    bool_first_success = first_trial(code)
  g = tokenize.generate_tokens(StringIO(code).readline)
  term = g.next()

  bool_finished = False
  while (not bool_finished):
    term_type = term[0]
    lineno = term[2][0] - 1
    posno = term[3][1] - 1
    if token.tok_name[term_type] in {"NUMBER", "STRING", "NEWLINE"}:
      tokenized_code.append(token.tok_name[term_type])
    elif not token.tok_name[term_type] in {"COMMENT", "ENDMARKER"} and len(term[1].strip()):
      candidate = term[1].strip()
      if candidate not in varnames:
        tokenized_code.append(candidate)
      else:
        tokenized_code.append("VAR")

    # fetch the next term
    bool_success_next = False
    while (not bool_success_next):
      try:
        term = g.next()
      except StopIteration:
        bool_finished = True
        break
      except:
        bool_failed_token = True
        print("Failed line: ")
        # print sys.exc_info()
        # tokenize the error line with wordpunct_tokenizer
        code_lines = code.split("\n")
        # if lineno <= len(code_lines) - 1:
        if lineno > len(code_lines) - 1:
          print sys.exc_info()
        else:
          failed_code_line = code_lines[lineno]  # error line
          print("Failed code line: %s" % failed_code_line)
          if posno < len(failed_code_line) - 1:
            print("Failed position: %d" % posno)
            failed_code_line = failed_code_line[posno:]
            tokenized_failed_code_line = wordpunct_tokenize(failed_code_line)  # tokenize the failed line segment
            print("wordpunct_tokenizer tokenization: ")
            print(tokenized_failed_code_line)
            # append to previous tokenizing outputs
            tokenized_code += tokenized_failed_code_line
          if lineno < len(code_lines) - 1:
            code = "\n".join(code_lines[lineno + 1:])
            g = tokenize.generate_tokens(StringIO(code).readline)
          else:
            bool_finished = True
            break
      else:
        bool_success_next = True

  return tokenized_code, bool_failed_var, bool_failed_token



def tokenize_sql_code(code, bool_remove_comment=True):
  """
  Best parsing for SQL code snippets.
  Credit to UW codenn project.

  Args:
    code: a string, a SQL code snippet.

  Returns:
    tokens: a list of tokens, where columns and tables are replaced with special token + id.

  """
  query = SqlTemplate(code, regex=True)
  typedCode = query.parseSql()
  tokens = [re.sub('\s+', ' ', x.strip()) for x in typedCode]

  if bool_remove_comment:
    tokens_remove_comment = []
    for token in tokens:
      if token[0:2] == "--":
        pass
      else:
        tokens_remove_comment.append(token)
    tokens = tokens_remove_comment

  return tokens, 0, 0

def tokenize_code_corpus(qid_to_code, pl):
  """ Tokenizing a code snippet into a list of tokens.
  Numbers/strings are replaced with NUMBER/STRING.
  Comments are removed. 

  (modified: replacing variable names with VAR)"""

  failed_token_qids = set()  # not tokenizable
  failed_var_qids = set() # not parsable to have vars

  qid_to_tokenized_code = dict()

  count = 0
  for qid, code in qid_to_code.items():
    count += 1
    if count % 1000 == 0:
      print count

    # unicode --> ascii
    code = code.encode("ascii", "ignore").strip()
    if len(code) == 0:
      tokenized_code = [""]

    else:
      if pl == "python":
        tokenized_code, bool_failed_var, bool_failed_token = tokenize_python_code(code)
      elif pl == "sql":
        tokenized_code, bool_failed_var, bool_failed_token = tokenize_sql_code(code)
      else:
        raise Exception("Invalid programming language! (Support python and sql only.)")

      if bool_failed_token:
        failed_token_qids.add(qid)
        print("failed tokenization qid: %s" % str(qid))
      if bool_failed_var:
        failed_var_qids.add(qid)

    sys.stdout.flush() # print info

    # save
    qid_to_tokenized_code[qid] = tokenized_code

  print("Total size: %d. Fails: %d." % (len(qid_to_tokenized_code), len(failed_token_qids)))

  return qid_to_tokenized_code, failed_var_qids, failed_token_qids

def main():
  def test_tokenize_code_corpus():
    code_corpus = {
    	"1": "def clamp(n, minn, maxn):\n  return max(min(maxn, n), minn)",
    	"2": "In [7]: df = DataFrame({'A' : list('aabbcd'), 'B' : list('ffghhe')})\n\nIn [8]:"
          " df\nOut[8]: \n   A  B\n0  a  f\n1  a  f\n2  b  g\n3  b  h\n4  c  h\n5  d  e\n\nIn "
          "[9]: df.dtypes\nOut[9]: \nA    object\nB    object\ndtype: object\n\nIn [10]: "
          "df.apply(lambda x: x.astype('category'))       \nOut[10]: \n   A  B\n0  a  f\n1  a  "
          "f\n2  b  g\n3  b  h\n4  c  h\n5  d  e\n\nIn [11]: df.apply(lambda x:"
          " x.astype('category')).dtypes\nOut[11]: \nA    category\nB    category\ndtype: object\n"}
    # code_corpus = {
    #   "1": u'| ID | COLOUR |\n|----|--------|\n|  1 |   Blue |\n|  4 |  Green |\n|  '
    #                     u'5 | Orange |\n|  6 |   Teal |\n|  3 | Yellow |\n|  2 |    Red |\n',
    #                "2": u'Contacts\n----------------------------------\nID          AuditLogID  CreatedOn\n'
    #                     u'----------- ----------- ----------\n10          1           2015-01-02\n11          3'
    #                     u'           2015-05-06\n\nAddresses\n----------------------------------\nID          '
    #                     u'AuditLogID  CreatedOn\n----------- ----------- ----------\n20          4           '
    #                     u'2014-02-01\n21          5           2010-01-01\n\nItems\n----------------------------------\n'
    #                     u'ID          AuditLogID  CreatedOn\n----------- ----------- ----------\n30          2           '
    #                     u'2015-03-04\n31          6           2011-03-04\n',
    #                "3": u'ID      STATUS  CONVERSATION_ID   MESSAGE_ID    DATE_CREATED\n3         2         '
    #                     u'2                95         May, 05 2012 \n2         2         1                87         '
    #                     u'March, 03 2012 \n',
    #   "4": "INSERT INTO tournTypes VALUES\n(1,2),\n(1,3),\n(2,3),\n(3,1)\n\nINSERT INTO leagueTypes VALUES\n(16,2,0), -- 16 teams, 2 divisions, teams only play within own division\n(8,1,0),\n(28,4,1)\n\nINSERT INTO playoffTypes VALUES\n(8,0), -- 8 teams, single elimination\n(4,0),\n(8,1)\n\nINSERT INTO Schedule VALUES\n('Champions league','2015-12-10','2016-02-10',1),\n('Rec league','2015-11-30','2016-03-04-,2)\n",
    #   "5": u'$videos = Carousel::find(2)->videos; //finds all videos associated with carousel having id of 2\n\nreturn $videos;\n',
    #   "6": u" SQL> create table mytbl (data_col varchar2(200));\n Table created\n SQL> insert into mytbl values('\u5728\u804c'); \n 1 row inserted.\n SQL> commit;\n Commit complete.\n SQL> select * from mytbl where data_col like '%\u5728\u804c%';\n DATA_COL                                                                                                                                                                                               \n -----------\n \u5728\u804c \n\n SQL> SELECT * FROM nls_database_parameters where parameter='NLS_CHARACTERSET';\n PARAMETER                      VALUE                                  \n ------------------------------ ----------------------------------------\n NLS_CHARACTERSET               AL32UTF8   \n",
    #   "7": "--test testtest2 test2"
    #   }
    tokenized_code, bool_failed_var, bool_failed_token = tokenize_code_corpus(code_corpus, "python")
    print(tokenized_code)

  test_tokenize_code_corpus()


if __name__ == "__main__":
  main()
