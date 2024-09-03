# end exploring, from now on this is in acutal training script
# create dataset
# TODO test sharded
import pandas as pd    
import transformers
from datasets import load_dataset, load_metric, Dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer



batch_size = 64
model_base = "/data/agent_h/llms/umt5-small"
model_name = "umt5-small-medium-title-generation-zh"
dataset_name = "/data/agent_h/umt5-small-news2016zh-tokens-full"
model_dir = f"/data/agent_h/checkpoints/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_base)
train_data = load_from_disk(dataset_name)

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_strategy="steps",
    logging_steps=1000,
    save_strategy="steps",
    save_steps=2000,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=100,
    predict_with_generate=True,
    bf16=True,
    report_to="tensorboard"
)
data_collator = DataCollatorForSeq2Seq(tokenizer)

def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_base)

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=train_data['train'],
    eval_dataset=train_data['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
)


trainer.train() 