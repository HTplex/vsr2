# load raw set
import pandas
raw_dataset_path = "/data/agent_h/datasets/medium_articles.csv"
tmp_df = pandas.read_csv(raw_dataset_path)
dicts = tmp_df.to_dict('records')

# explore data format
from pprint import pprint
print(len(dicts))
sample_dict = dicts[0].copy()
sample_dict['text'] = sample_dict['text'][:200]
pprint(dicts[0])

# use dataset lib, https://huggingface.co/docs/datasets/en/loading
# best way would be raw_data -> process into train -> save as csv or json chunks -> load as dataset
import transformers
from datasets import load_dataset, load_metric, Dataset

# medium_datasets = load_dataset("csv",
#                                data_files=raw_dataset_path)

# before using from_list,
# need to make sure each key in the list has the same type of value
timestamp_types = set([type(x['timestamp']) for x in dicts])
print(timestamp_types)
# clean up data
filtered_data = []
for data_line in dicts:
    valid = True
    for key,item in data_line.items():
        if type(item) != str:
            valid = False
    if valid:
        filtered_data.append(data_line)
print(len(filtered_data))
medium_dataset = Dataset.from_list(filtered_data[:])

# process data for training https://huggingface.co/docs/datasets/en/process
# split first so there's no leaking
medium_dataset = medium_dataset.filter(
    lambda example: (len(example['text']) >= 500) and
    (len(example['title']) >= 20)
)
medium_dataset = medium_dataset.train_test_split(test_size=1000)

# process for training
import nltk
import string
# nltk.download('punkt')
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("/data/agent_h/llms/umt5-small")
tokenizer = AutoTokenizer.from_pretrained("/data/agent_h/llms/umt5-small")

prefix = "summarize: "
max_input_length = 1024
max_target_length = 128


def clean_text(text):
    """
    add \n to sentences, remove title
    """
    sentences = nltk.sent_tokenize(text.strip())
    sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
    sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                 if len(sent) > 0 and
                                 sent[-1] in string.punctuation]
    text_cleaned = "\n".join(sentences_cleaned_no_titles)
    return text_cleaned

#pprint(clean_text(medium_dataset['train'][0]['text']))

def preprocess_data(examples):
    "turn into tokens for labels and input_ids"
    texts_cleaned = [clean_text(text) for text in examples["text"]]
    inputs = [prefix + text for text in texts_cleaned]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["title"], max_length=max_target_length, 
                           truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                      for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# examples = medium_dataset['test'][:100]
# tmp_data = preprocess_data(examples)
# print(examples['text'][8])
# tokenizer.decode(tmp_data['input_ids'][8])
tokenized_datasets = medium_dataset.map(preprocess_data,
                                        batched=True)
# start training

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

batch_size = 32
base_model = "/data/agent_h/llms/umt5-small"
model_name = "umt5-small-medium-title-generation"
model_dir = f"/data/agent_h/checkpoints/{model_name}"

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    # fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)
data_collator = DataCollatorForSeq2Seq(tokenizer)
metric = load_metric("rouge")

def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(base_model)

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()
