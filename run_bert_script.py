import pandas as pd
import torch 
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD
import re

FILEPATH = './data/my_train_NER.csv'
LEARNING_RATE = 5e-3
EPOCHS = 1
BATCH_SIZE = 4

df = pd.read_csv(FILEPATH)
print(f'df contains {df.shape[0]} sentences')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

df = df[:500]

unique_labels = {'B-event',
 'B-geo',
 'B-gpe',
 'B-obj',
 'B-org',
 'B-per',
 'B-time',
 'I-event',
 'I-geo',
 'I-gpe',
 'I-obj',
 'I-org',
 'I-per',
 'I-time',
 'O',
 'nan'}

labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
ids_to_labels = {v: k for v, k in enumerate(unique_labels)}

df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                            [int(.8 * len(df)), int(.9 * len(df))])

label_all_tokens = False

def align_label(texts, labels):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df):

        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels

df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                            [int(.8 * len(df)), int(.9 * len(df))])

class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output

def get_clean_word(inword):
  if  len(str(inword)) > 1:
    if str(inword) == '--':
      stri = '-'
    else:
      stri = re.sub("-|'|_|~", '', str(inword))
      stri = stri.replace('.', '')
      stri = stri.replace(',', '')
  else:
    stri = str(inword)
  return stri

def get_preds(model, epoch):
  TESTFILENAMES = ['Test1NER.csv', 'Test2NER.csv']
  header_names = ['Sentences', 'Word', 'Predicted']
  for TESTFILENAME in TESTFILENAMES:
    test_df = pd.read_csv('./data/'+TESTFILENAME, sep=';', encoding= 'unicode_escape', names=header_names)
    test_df = test_df.replace('...','.')
    words = test_df['Word'].values

    sentence_column = test_df['Sentences'].values
    sentence_column = [str(cell) for cell in sentence_column]

    sentences = []
    sentence_tags = []

    for i in range(len(sentence_column)):
        if sentence_column[i] != 'nan':
            if i != 0 :sentences.append(sentence)
            sentence = []
            sentence.append(get_clean_word(words[i]))
        else:
            sentence.append(get_clean_word(words[i]))
    sentences.append(sentence)
    list_of_string_sentences = [[' '.join(sentence)] for sentence in sentences]
    list_of_string_sentences = [item for sublist in list_of_string_sentences for item in sublist]

    predictions = [evaluate_one_text(model, sentenc) for sentenc in list_of_string_sentences]

    flat_list = [item for sublist in predictions for item in sublist]
    
    test_df['Predicted']=flat_list
    test_df.to_csv('./res/'+TESTFILENAME+'_'+str(epoch), index=False, sep=';')

def train_loop(model, df_train, df_val, LEARNING_RATE, EPOCHS, BATCH_SIZE):

    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][train_label[i] != -100]
              label_clean = train_label[i][train_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_train += acc
              total_loss_train += loss.item()

            loss.backward()
            optimizer.step()
            model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][val_label[i] != -100]
              label_clean = val_label[i][val_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_val += acc
              total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)
        print('Example is:\nText: Bill Gates is the founder of Microsoft\n', evaluate_one_text(model, 'Bill Gates is the founder of Microsoft'))
        

        get_preds(model, epoch_num)
        print('predicting done.')
        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}')

def align_word_ids(texts):
  
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def evaluate_one_text(model, sentence):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    return prediction_label


model = BertModel()
train_loop(model, df_train, df_val, LEARNING_RATE, EPOCHS, BATCH_SIZE)
