# Installing necessary libraries for dataset handling and transformer models

# !pip install datasets
# !pip install accelerate -U
# !pip install -U transformers

# Importing the load_dataset function from the Hugging Face's datasets library
from datasets import load_dataset

# Importing respective tokenizers for transformer models from the Hugging Face's transformers library
from transformers import DistilBertTokenizer
# from transformers import BertTokenizer
# from transformers import AlbertTokenizer

# Importing respective dataset
dataset = load_dataset("dair-ai/emotion")
# dataset = load_dataset("tweet_eval", "irony")
print(dataset)

# Calculating the number of unique labels/classes in the training dataset (either 2 (irony) or 6 (emotion))
num_of_labels = dataset['train'].features['label'].num_classes
print(num_of_labels)

# Initiailizaing respective tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# Defining a function to tokenize the input text
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Appling the respective tokenization function to the entire dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# In some part of experimentations, subsets (30%/50%) of test, validation, and trainig data were chosen for quicker experimentation and evaluation

# test_dataset = tokenized_irony_dataset["test"]
# test_subset = test_dataset.shuffle(seed=42).select(range(int(0.5 * len(test_dataset))))

# train_dataset = tokenized_irony_dataset["train"]
# train_subset = train_dataset.shuffle(seed=42).select(range(int(0.5 * len(train_dataset))))

# valid_dataset = tokenized_irony_dataset["validation"]
# valid_subset = valid_dataset.shuffle(seed=42).select(range(int(0.5 * len(valid_dataset))))


# Importing the respective class for sequence classification models and training utilities from the transformers library
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
# from transformers import BertForSequenceClassification, Trainer, TrainingArguments
# from transformers import AlbertForSequenceClassification, Trainer, TrainingArguments

import numpy as np # Importing numpy for numerical operations
from sklearn.metrics import accuracy_score, confusion_matrix # Importing accuracy_score and confusion_matrix for evaluation metrics

# Choosing respective pre-trained model for the task. 
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_of_labels)
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_of_labels)
# model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=num_of_labels)

# Setting up the training arguments specifying how the model should be fine-tuned
training_args = TrainingArguments(
  output_dir="./results", # Directory for saving outputs
  num_train_epochs=2,  # The number of epochs to train the model.
  learning_rate=2e-5, # The learning rate for the optimizer.
  per_device_train_batch_size=16, # Batch size for training.
  per_device_eval_batch_size=16, # Batch size for evaluation
  weight_decay=0.01, # Adds L2 regularization to help prevent the model from overfitting
  evaluation_strategy="epoch", # Evaluation is done at the end of each epoch
  save_strategy="epoch",  # Save the model after each epoch
#   load_best_model_at_end=True  # Load the best model when finished training
)

# Initializing the Trainer with the model, training arguments, datasets, and metric computation
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_dataset["train"],
  eval_dataset=tokenized_dataset["validation"],
  compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}
)

# Starting the training process
trainer.train()

# After training, evaluating the model's performance on the test dataset
eval_results = trainer.evaluate(tokenized_dataset["test"])
# eval_results = trainer.evaluate(test_subset)
print("Evaluation Results:", eval_results)

# After training, evaluating the model's performance on the train dataset
train_eval_results = trainer.evaluate(tokenized_dataset["train"])
# train_eval_results = trainer.evaluate(train_subset)
print("Training Evaluation Results:", train_eval_results)

# After training, evaluating the model's performance on the validation dataset
validation_eval_results = trainer.evaluate(tokenized_dataset["validation"])
# validation_eval_results = trainer.evaluate(valid_subset)
print("Validation Evaluation Results:", validation_eval_results)


# After training and evaluation, making predictions on the test dataset to assess the model's performance
predictions = trainer.predict(tokenized_dataset["test"])
# predictions = trainer.predict(test_subset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Generating a confusion matrix to evaluate the model's performance in tweet/irony Dataset (as requested on assignment)

# true_labels = tokenized_dataset["test"]['label']
# confusion_mat = confusion_matrix(true_labels, predicted_labels)
# print("Confusion Matrix:")
# print(confusion_mat)

# For better clarity, separetaly identifying TN, FP, FN, TP 

# TN, FP, FN, TP = confusion_mat.ravel()
# print(f"True Negatives (TN): {TN}")
# print(f"False Positives (FP): {FP}")
# print(f"False Negatives (FN): {FN}")
# print(f"True Positives (TP): {TP}")

# This section performs a qualitative analysis of the respective model's predictions by randomly selecting 5 samples from the respective test dataset. 
# For each selected sample, it decodes the tokenized text back into readable text, retrieves the true label from  the dataset, and matches it with the corresponding predicted label generated by the model earlier.
# Each sample's text, true label, and predicted label are compiled into a dictionary and appended to a list.

# from random import sample
# # test_samples = test_subset
# test_samples = tokenized_dataset["test"]
# all_indices = list(range(len(test_samples)))
# selected_indices = sample(all_indices, 5)
# samples = test_samples.select(selected_indices)

# analysis = []
# for i in range(5):
#     sample = samples[i]
#     text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
#     true_label = sample['label']
#     predicted_label = predicted_labels[i]
#     analysis.append({
#         "Text": text,
#         "True Label": true_label,
#         "Predicted Label": predicted_label,
#     })

# for i in analysis:
#     print(i)

