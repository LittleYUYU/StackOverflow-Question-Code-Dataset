# code_answer_inference
Code and data for "StaQC: A Systematically Mined Question-Code Dataset from Stack Overflow (WWW'18)"

## Paper
TBA

----------------
## StaQC dataset
<table>
  <tr>
    <td></td>
    <td colspan="2"><strong>#of QC pair</strong></td>
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

### Multi-code answer posts & manual annotations
The question-code pairs mined or manually annotated from multi-code answer posts can be found: [Python](final_collection/python_multi_code_iids.txt) and [SQL](final_collection/sql_multi_code_iids.txt). Each line in the file corresponds to one code snippet, which can be paired with its question.
<br> **Format**: Each code snippet is identified by `(question id, code snippet index)`, where the `code snippet index` refers to the index (starting from 0) of the code snippet in the accepted answer post of this question. For example, `(5996881, 0)` refers to the first code snippet in the accepted answer post of the [question](https://stackoverflow.com/a/5996949) with id "5996881", which can be paired with its question "How to limit a number to be within a specified range? (Python)".
<br> **Source data**: [Python Pickle](https://docs.python.org/2/library/pickle.html) files. Please open with `pickle.load(open(filename))`.
- Code snippets for [Python](annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_iid_to_code.pickle) and [SQL](annotation_tool/data/code_solution_labeled_data/source/sql_how_to_do_it_by_classifier_multiple_iid_to_code.pickle): A dict of {(question id, code index): code snippet}.
- Question titles for [Python](annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_by_classifier_multiple_qid_to_title.pickle) and [SQL](annotation_tool/data/code_solution_labeled_data/source/sql_how_to_do_it_by_classifier_multiple_qid_to_title.pickle): A dict of {question id: question title}.

#### Manual annotations
Human annotations can be found: [Python](annotation_tool/crowd_sourcing/python_annotator/all_agreed_iid_to_label.pickle) and [SQL](annotation_tool/crowd_sourcing/sql_annotator/all_agreed_iid_to_label.pickle). Both are pickle files.

### Single-code answer posts
The code snippets and question titles for single-code answer posts are also provided:
- Code snippets for [Python](annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle) and for [SQL]((annotation_tool/data/code_solution_labeled_data/source/sql_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_code.pickle)): A dict of {question id: accepted code snippet}.
- Question titles for [Python](annotation_tool/data/code_solution_labeled_data/source/python_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_title.pickle) and [SQL](annotation_tool/data/code_solution_labeled_data/source/sql_how_to_do_it_qid_by_classifier_unlabeled_single_code_answer_qid_to_title.pickle): A dict of {question id: question title}.

---------------
## Run the code
- Requirements: Tensorflow
- Raw Stack Overflow (SO) dump can be found: https://archive.org/details/stackexchange. We use SO dump collected from 07/31/2008 to 06/12/2016 in experiments.
For running our model BiV-HNN directly on the processed data, please see [**Run BiV-HNN**](#run-biv-hnn).

### Raw data processing

### How-to-do-it question type classifier

### Data processing

### Run BiV-HNN
We provide processed training/validation/testing files in our experiments. Before running, please unzip the word embedding files (code_word_embedding.z01/z02/zip) under this [folder](data/data_hnn/python/train/). 

`cd BiV_HNN`

**Train**:
- For Python data:<br>
`python run.py --train --train_setting=1 --text_model=1 --code_model=1 --query_model=1 --text_model_setting="64-150-24379-0-1-0-1" --code_model_setting="64-150-218900-0-1-0-1" --text_model_setting="64-150-24379-0-1-0-1" --keep_prob=0.5`
- For SQL data:<br>
`python run.py --train --train_setting=2 --text_model=1 --code_model=1 --query_model=1 --text_model_setting="64-150-13698-0-1-0-1" --code_model_setting="64-150-33192-0-1-0-1" --text_model_setting="64-150-13698-0-1-0-1" --keep_prob=0.7`

The above program trains the `BiV-HNN` model. It will print the model's learning process on the training set, and its performance on the validation set and the testing set. 

For training `Text-HNN`, set:<br>
`--code_model=0 --query_model=0 --code_model_setting=None --query_model_setting=None`<br>
to dismiss the code and query modeling.

For training `Code-HNN`, set:<br>
`--text_model=0 --text_model_setting=None`<br>
to dismiss the text modeling.

**Test**:<br>
You may revise the `test` function in `run.py` for testing other datasets, and run the above command (Note: replace `--train` with `--test`). 
