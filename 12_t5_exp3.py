# end exploring, from now on this is in acutal training script
# create dataset
# TODO test sharded
import pandas as pd    
import transformers
from datasets import load_dataset, load_metric, Dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer



batch_size = 12
model_base = "/data/agent_h/vsr2/llms/umt5-xl"
model_name = "umt5-xl-medium-title-generation-zh"
dataset_name = "/data/agent_h/vsr2/datasets/umt5-small-news2016zh-tokens-full"
model_dir = f"/data/agent_h/vsr2/checkpoints/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_base)
train_data = load_from_disk(dataset_name)

deep_speed_config = {
            "zero_optimization": {
                      "stage": 1
                },
            "gradient_accumulation_steps": 80,
            "train_micro_batch_size_per_gpu": 12,
            "gradient_clipping": 1.0,
            "bf16": {
                "enabled": True
                }
            }


args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=10,
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=80,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1000,
    predict_with_generate=True,
    bf16=True,
    deepspeed=deep_speed_config,
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
