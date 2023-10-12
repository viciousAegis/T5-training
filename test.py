# %%
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets, load_metric
import datetime
import numpy as np

# load model from checkpoint and tokenizer
MODEL_PATH = "./models/flan-t5-base-squad-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# test model
# %%
dataset = load_dataset("squad_v2")

# %%
dataset['train'][0]
dataset['validation'][0]

# %%
# preprocess

# have at most 1 answer for each question

def squad_preprocess(example):
    if(len(example['answers']['answer_start'])==0):
        example['isAnswerable'] = "No"
    else:
        example['isAnswerable'] = "Yes"
    # remove answers column if there
    del example['answers']
    del example['title']
    del example['id']
    return example

# %%
dataset = dataset.map(squad_preprocess)
dataset['train'][0]
# %% 

# load local datasets and seperate into train and validation
pos_samples_dataset = load_dataset('json', data_files='./data/positive_samples.json')
neg_samples_dataset = load_dataset('json', data_files='./data/negative_samples.json')

# take some samples and move them to validation
# take 10% of positive samples and 10% of negative samples
num_pos = len(pos_samples_dataset['train'])
num_neg = len(neg_samples_dataset['train'])

# filter out validation samples from train
pos_samples_dataset['validation'] = pos_samples_dataset['train'].filter(lambda example, idx: idx % 10 == 0, with_indices=True)
neg_samples_dataset['validation'] = neg_samples_dataset['train'].filter(lambda example, idx: idx % 10 == 0, with_indices=True)

# remove validation samples from train
pos_samples_dataset['train'] = pos_samples_dataset['train'].filter(lambda example, idx: idx % 10 != 0, with_indices=True)
neg_samples_dataset['train'] = neg_samples_dataset['train'].filter(lambda example, idx: idx % 10 != 0, with_indices=True)

# %%
def preprocess_NQ_dataset(example, isPositive):
    # rename query to question
    example['question'] = example['query']
    # remove query column
    del example['query']
    
    # remove genai_response column
    del example['genai_response']
    
    example['isAnswerable'] = "Yes" if isPositive else "No"
    
    return example
# %%
pos_samples_dataset = pos_samples_dataset.map(lambda example: preprocess_NQ_dataset(example, True))
neg_samples_dataset = neg_samples_dataset.map(lambda example: preprocess_NQ_dataset(example, False))

# %%
# load subtl dataset
subtl_pos_dataset = load_dataset('json', data_files='./data/positive_subtl_dataset.json')
subtl_neg_dataset = load_dataset('json', data_files='./data/negative_subtl_dataset.json')

# take 10% of positive samples and 10% of negative samples
num_pos = len(subtl_pos_dataset['train'])
num_neg = len(subtl_neg_dataset['train'])

# filter out validation samples from train
subtl_pos_dataset['validation'] = subtl_pos_dataset['train'].filter(lambda example, idx: idx % 10 == 0, with_indices=True)
subtl_neg_dataset['validation'] = subtl_neg_dataset['train'].filter(lambda example, idx: idx % 10 == 0, with_indices=True)

# remove validation samples from train
subtl_pos_dataset['train'] = subtl_pos_dataset['train'].filter(lambda example, idx: idx % 10 != 0, with_indices=True)
subtl_neg_dataset['train'] = subtl_neg_dataset['train'].filter(lambda example, idx: idx % 10 != 0, with_indices=True)


# %%
def preprocess_subtl_dataset(example, isPositive):
    # rename query to question
    example['question'] = example['query']
    # remove query column
    del example['query']
    
    del example['id']
    
    # remove genai_response column
    del example['genai_response']
    
    example['isAnswerable'] = "Yes" if isPositive else "No"
    
    return example

# %%
subtl_pos_dataset = subtl_pos_dataset.map(lambda example: preprocess_subtl_dataset(example, True))
subtl_neg_dataset = subtl_neg_dataset.map(lambda example: preprocess_subtl_dataset(example, False))

subtl_pos_dataset

# %%
# combine all datasets

# dataset['train'] = concatenate_datasets([
#     dataset['train'],
#     pos_samples_dataset['train'],
#     neg_samples_dataset['train'],
#     subtl_pos_dataset['train'],
#     subtl_neg_dataset['train'],
# ])

# dataset['validation'] = concatenate_datasets([
#     dataset['validation'],
#     pos_samples_dataset['validation'],
#     neg_samples_dataset['validation'],
#     subtl_pos_dataset['validation'],
#     subtl_neg_dataset['validation'],
# ])

dataset['validation'] = concatenate_datasets([
    subtl_pos_dataset['validation'],
    subtl_neg_dataset['validation'],
    subtl_pos_dataset['train'],
    subtl_neg_dataset['train'],
])  
# %%

# convert dataset to pytorch tensors using the tokenizer
def convert_to_features(example_batch):
    # Tokenize contexts and questions (as pairs of inputs)
    prefix = "Is this question answerable given the context: "
    
    encodings = tokenizer(
        [prefix + question for question in example_batch['question']],
        example_batch['context'],
        truncation='only_second',
        padding='max_length',
        max_length=512,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    
    # Tokenize answers
    answers = tokenizer(
        example_batch['isAnswerable'],
        truncation=True,
        padding='max_length',
        max_length=8,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    # replace -100 in the labels as we can't decode them.
    labels = answers['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100

    encodings['labels'] = labels

    return encodings

# %%
def postprocess_text(preds, labels):
    preds = [pred.replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '') for pred in preds]
    labels = [[label.replace(tokenizer.pad_token, '').replace(tokenizer.eos_token, '')] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    print(eval_preds)
    preds, labels = eval_preds
    
    acc = load_metric('accuracy')
    precision = load_metric('precision')
    recall = load_metric('recall')
    f1 = load_metric('f1')
    
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    # map yes/no to 0/1
    
    # convert to lowercase
    decoded_labels = [label[0].lower() for label in decoded_labels]
    decoded_preds = [pred.lower() for pred in decoded_preds]
    
    decoded_labels = [0 if label == 'no' else 1 for label in decoded_labels]
    decoded_preds = [0 if pred == 'no' else 1 for pred in decoded_preds]
   
    # print labels and preds in a table
    print("Labels\tPreds")
    for i in range(len(decoded_labels)):
        print(f"{decoded_labels[i]}\t{decoded_preds[i]}")
   
    accuracy = acc.compute(predictions=decoded_preds, references=decoded_labels)
    precision = precision.compute(predictions=decoded_preds, references=decoded_labels)
    recall = recall.compute(predictions=decoded_preds, references=decoded_labels)
    f1 = f1.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {
        'accuracy': accuracy['accuracy'],
        'precision': precision['precision'],
        'recall': recall['recall'],
        'f1': f1['f1']
    }

# %%
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# define training args
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    overwrite_output_dir=True,
    save_total_limit=2,
    fp16=True,
    learning_rate=1e-4,
    num_train_epochs=2,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    evaluation_strategy='epoch',
    save_strategy='epoch'
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=dataset['validation']
    .map(convert_to_features, batched=True),
    compute_metrics=compute_metrics
)

# evaluate model
res = trainer.evaluate()
print(res)
# %%

# print results to file
with open('./results.txt', 'w') as f:
    f.write(f"Run: {datetime.datetime.now()}\n")
    f.write("====================================\n")
    f.write("Dataset details:\n")
    f.write("Name: Subtl Dataset\n")
    f.write(f"Number of total samples: {len(dataset['validation'])}\n")
    
    # write formatted results
    f.write("====================================\n")
    f.write("Results:\n")
    f.write(f"Accuracy: {res['eval_accuracy']}\n")
    f.write(f"Precision: {res['eval_precision']}\n")
    f.write(f"Recall: {res['eval_recall']}\n")
    f.write(f"F1: {res['eval_f1']}\n")
    f.write(f"Runtime: {res['eval_runtime']}\n")
    f.write(f"Samples per second: {res['eval_samples_per_second']}\n")
    f.write(f"Steps per second: {res['eval_steps_per_second']}\n")

# %%
