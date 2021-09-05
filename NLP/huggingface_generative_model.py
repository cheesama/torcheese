from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2TokenizerFast, PreTrainedTokenizerFast
from transformers import Trainer, TrainingArguments
from tokenizers import BertWordPieceTokenizer, CharBPETokenizer

import torch

config = GPT2Config()

# for fine-tuning
#model = GPT2LMHeadModel.from_pretrained('gpt2-large') 
#model = GPT2LMHeadModel.from_pretrained('gpt2') 
#tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)

# for pre-training
model = GPT2LMHeadModel(config)

# LMHeadModel forward example
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
logits = outputs.logits

# using transformers trainer module(hard to customize model internal)
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset(custom dataset)
    eval_dataset=val_dataset             # evaluation dataset(custom dataset)
)

trainer.train()


