# StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow

## 1. StaQC dataset

### 1.1 Introduction
StaQC (**Sta**ck Overflow **Q**uestion-**C**ode pairs) contains around **148K** Python and **120K** SQL domain question-code pairs, which are automatically mined from [Stack Overflow](https://stackoverflow.com/) using a Bi-View Hierarchical Neural Network, as described in the paper "[StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow](http://web.cse.ohio-state.edu/~sun.397/Publications.htm)" (WWW'18).

StaQC is collected from three sources: multi-code answer posts, single-code answer posts, and manual annotations on multi-code answer posts:
<table>
  <tr>
    <td></td>
    <td colspan="2"><strong>#of question-code pair</strong></td>
  </tr>
  <tr>
    <td><strong>Source</strong></td>
    <td><strong>Python</strong></td>
    <td><strong>SQL</strong></td>
  </tr>
  <tr>
    <td>Multi-Code Answer Posts</td>
    <td>60,083</td>
    <td>41,826</td>
  </tr>
  <tr>
    <td>Single-Code Answer Posts</td>
    <td>85,294</td>
    <td>75,637</td>
  </tr>
  <tr>
    <td>Manual Annotation</td>
    <td>2,169</td>
    <td>2,056</td>
  </tr>
  <tr>
    <td><strong>Sum</strong></td>
    <td><strong>147,546</strong></td>
    <td><strong>119,519</strong></td>
  </tr>
</table>

### 1.2 Multi-code answer posts & manual annotations
A *Multi-code answer post* is an (accepted) answer post that contains multiple code snippets, some of which may not be a *standalone* code solution to the question (see Section 1 in [paper](http://web.cse.ohio-state.edu/~sun.397/Publications.htm)). For example, in [this multi-code answer post](https://stackoverflow.com/a/5996949), the third code snippet is not a code solution to the question "How to limit a number to be within a specified range? (Python)".

The question-code pairs automatically mined or manually annotated from multi-code answer posts can be found here: [Python](final_collection/python_multi_code_iids.txt) and [SQL](final_collection/sql_multi_code_iids.txt). 
<br> **Format**: Each line corresponds to one code snippet, which can be paired with its question. The code snippet is identified by `(question id, code snippet index)`, where the `code snippet index` refers to the index (starting from 0) of the code snippet in the accepted answer post of this question. For example, `(5996881, 0)` refers to the first code snippet in the accepted answer post of the [question](https://stackoverflow.com/a/5996949) with id "5996881", which can be paired with its question "How to limit a number to be within a specified range? (Python)".
<br> **Source data**: [Python Pickle](https://docs.python.org/2/library/pickle.html) files. Please open with `pickle.load(open(filename))`.
- Code snippets for [Python](annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle) and [SQL](annotation_tool/data/code_solution_labeled_data/source/sql_how_to_do_it_by_classifier_multiple_iid_to_code.pickle): A dict of {(question id, code index): code snippet}.
- Question titles for [Python](annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_qid_to_title.pickle) and [SQL](annotation_tool/data/code_solution_labeled_data/source/sql_how_to_do_it_by_classifier_multiple_qid_to_title.pickle): A dict of {question id: question title}.

### 1.3 Single-code answer posts
A *Single-code answer post* is an (accepted) answer post that contains only one code snippet. We pair such code snippet with the question title as a question-code pair.

**Source data**: [Python Pickle](https://docs.python.org/2/library/pickle.html) files. Please open with `pickle.load(open(filename))`.
- Code snippets for [Python](annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle) and for [SQL]((annotation_tool/data/code_solution_labeled_data/source/sql_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle)): A dict of {question id: accepted code snippet}.
- Question titles for [Python](annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_title.pickle) and [SQL](annotation_tool/data/code_solution_labeled_data/source/sql_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_title.pickle): A dict of {question id: question title}.


## 2. Software

### 2.1 Prerequisite
- Python 2.7
- Tensorflow (1.0.1 or later)
- [Raw Stack Overflow (SO) dump](https://archive.org/details/stackexchange) or [our processed data](data/data_hnn).

### 2.2 Manual annotations
Human annotations can be found: [Python](annotation_tool/crowd_sourcing/python_annotator/all_agreed_iid_to_label.pickle) and [SQL](annotation_tool/crowd_sourcing/sql_annotator/all_agreed_iid_to_label.pickle). Both are pickle files.

### 2.3 How-to-do-it question type classifier

### 2.4 Data processing

### 2.5 Run BiV-HNN
We provide processed training/validation/testing files in our experiments [here](data/data_hnn/). 

1. Before running, please unzip the word embedding files for Python (code_word_embedding.gz*) following:<br>
`cd data/data_hnn/python/train/`<br>
`cat code_word_embedding.gza* | zcat > rnn_partialcontext_word_embedding_code_150.pickle`<br>
`rm code_word_embedding.gza*`<br>
then go back the code dir:<br>
`cd ../../../../BiV_HNN/`.

   No other operations demanded for SQL data.

2. Train:<br>
   For Python data:<br>
   ```
   python run.py --train --train_setting=1 --text_model=1 --code_model=1 --query_model=1 --text_model_setting="64-150-24379-0-1-0-1" --code_model_setting="64-150-218900-0-1-0-1" --query_model_setting="64-150-24379-0-1-0-1" --keep_prob=0.5
   ```
  
   For SQL data:<br>
   ```
   python run.py --train --train_setting=2 --text_model=1 --code_model=1 --query_model=1 --text_model_setting="64-150-13698-0-1-0-1" --code_model_setting="64-150-33192-0-1-0-1" --query_model_setting="64-150-13698-0-1-0-1" --keep_prob=0.7
   ```

   The above program trains the `BiV-HNN` model. It will print the model's learning process on the training set, and its performance on the validation set and the testing set. 

   For training `Text-HNN`, set:<br>
   `--code_model=0 --query_model=0 --code_model_setting=None --query_model_setting=None`
   to dismiss the code and query modeling.

   For training `Code-HNN`, set:<br>
   `--text_model=0 --text_model_setting=None`<br>
   to dismiss the text modeling.

3. Test:<br>
You may revise the `test` function in `run.py` for testing other datasets, and run the above command (Note: replace `--train` with `--test`). 

## 3. Cite
If you use the dataset or the code in your research, please cite the following paper:
> StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow<br>
> Ziyu Yao, Daniel S. Weld, Wei-Peng Chen, Huan Sun

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
