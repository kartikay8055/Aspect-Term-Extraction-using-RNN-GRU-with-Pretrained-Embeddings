import nltk
import json
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from conlleval import evaluate 


nltk.download('punkt_tab')

def load_data(file_path):
  with open(file_path,'r') as file:
    data = json.load(file)
  return data

def get_bio_tags(sentence,aspect_terms):
  tokens = re.findall(r"\w+|[^\w\s]", sentence) #tokenizing the sentence into terms
  labels = ['O']*len(tokens)
  for aspect in aspect_terms:
    term = aspect['term']
    term_tokens = re.findall(r"\w+|[^\w\s]", term)
    for i in range(len(tokens)-len(term_tokens)+1):
      if tokens[i:i+len(term_tokens)]==term_tokens:
        labels[i] = "B"
        for j in range(1,len(term_tokens)):
          labels[i+j]="I"

  return labels,tokens



def preprocessed_data(input_file,output_file):
 
 
  data = load_data(input_file)
  preprocessed_list = []
  for item in tqdm(data,desc = f"Preprocessing(input_file)"):
    sentence = item['sentence']
    aspect_terms = item.get('aspect_terms',[]) 
    labels,tokens = get_bio_tags(sentence,aspect_terms)

    processed_entry ={
      'sentence':sentence,
      'tokens' : tokens,
      'labels' : labels,
      'aspect_terms' : [aspect['term'] for aspect in aspect_terms]

    }
    preprocessed_list.append(processed_entry)
    with open(output_file , 'w') as file:
      json.dump(preprocessed_list,file)
      file.write("\n\n")  # Adds a newline gap between entries
  print(f"preprocessed_data saved to {output_file}")



def build_vocabulary(dataset):
  vocab = set()
  for part in dataset:
    vocab.update(part['tokens'])
  vocab_list = ["<unk>","<pad>"]+sorted(vocab)
  vocab_dict = {word: idx for idx, word in enumerate(vocab_list)}  # Convert to dictionary
  return vocab_dict

def save_embeddings_pickle(embeddings, file_path):
    """Saves word embeddings in a pickle file for faster access."""
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {file_path}")

def load_embeddings(file_path, embedding_dim=100, pickle_path=None):
    """Loads embeddings from a text file or a pickle file if available."""
    if pickle_path and os.path.exists(pickle_path):
        print(f"Loading embeddings from cache: {pickle_path}")
        with open(pickle_path, "rb") as f:
            return pickle.load(f)

    print(f"Loading embeddings from text file: {file_path} (this may take time)...")
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector

    if pickle_path:
        save_embeddings_pickle(embeddings, pickle_path)

    return embeddings


def encode_sentence(sentence, vocab):
    return [vocab[word] if word in vocab else vocab["<unk>"] for word in sentence]

def prepare_data(dataset, vocab):
    sentences = []
    labels = []
    BIO_LABELS = {"O": 0, "B": 1, "I": 2}

    for part in dataset:
        encoded_sentence = torch.tensor(encode_sentence(part['tokens'], vocab))  # Convert words to indices
        encoded_labels = torch.tensor([BIO_LABELS[label] for label in part['labels']])  # Convert labels to numbers
        
        sentences.append(encoded_sentence)
        labels.append(encoded_labels)

    return DataLoader(TensorDataset(
        torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=vocab["<pad>"]),
        torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)  # Padding label = 0 (O)
    ), batch_size=32, shuffle=True)

  
class AspectTermExtractor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings, model_type="RNN", dropout_rate=0.4):
        super(AspectTermExtractor, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.hidden_dim = hidden_dim

        if model_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)  # Adding dropout
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        out = self.dropout(rnn_out)  # Apply dropout after RNN
        out = self.fc(out)
        return F.log_softmax(out, dim=2)

# Evaluate Model using conlleval
def calculate_f1_score(model, data_loader, vocab):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, ground_truths = [], []

    idx2label = {0: "O", 1: "B-TERM", 2: "I-TERM"}

    with torch.no_grad():
        for sentences, labels in data_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            outputs = model(sentences)

            predicted_labels = torch.argmax(outputs, dim=2).cpu().numpy()
            actual_labels = labels.cpu().numpy()

            for i in range(sentences.shape[0]):
                sentence_length = (labels[i] != 0).sum().item()  # Ignore padding tokens
                
                gold = [idx2label[actual_labels[i][j]] for j in range(sentence_length)]
                pred = [idx2label[predicted_labels[i][j]] for j in range(sentence_length)]

                ground_truths.extend(gold)  # Used .extend() to flatten the list
                predictions.extend(pred)  # Used .extend() to flatten the list

    # Compute chunk-level F1-score (conlleval)
    precision, recall, f1_chunk = evaluate(ground_truths, predictions)

    # Compute tag-level performance (token-level accuracy)
    correct_tags = sum([1 for g, p in zip(ground_truths, predictions) if g == p])
    total_tags = len(ground_truths)
    tag_accuracy = (correct_tags / total_tags) * 100 if total_tags > 0 else 0

    print(f"Chunk-Level - Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1-score: {f1_chunk:.2f}%")
    print(f"Tag-Level - Accuracy: {tag_accuracy:.2f}%")

    return f1_chunk, tag_accuracy



def train_model(model, train_loader, val_loader, vocab, model_name):
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignores padding tokens during loss computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    best_f1_chunk = 0  
    train_losses, val_losses = [], []

    # Train for 10 epochs
    for epoch in range(10):
        model.train()
        total_loss = 0

        for sentences, labels in train_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(sentences)
            loss = loss_fn(outputs.view(-1, outputs.shape[2]), labels.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevents gradient explosion
            
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        # Validation Loss Calculation (after every epoch)
        val_loss = 0
        with torch.no_grad():
            for sentences, labels in val_loader:
                sentences, labels = sentences.to(device), labels.to(device)
                outputs = model(sentences)
                loss = loss_fn(outputs.view(-1, outputs.shape[2]), labels.view(-1))
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}, Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

    # Now evaluate the model on the validation set after training is complete
    f1_chunk, tag_accuracy = calculate_f1_score(model, val_loader, vocab)

    # Save the model if it provides the best chunk-level F1 score
    if f1_chunk > best_f1_chunk:
        best_f1_chunk = f1_chunk
        torch.save(model.state_dict(), model_name)
        print(f"Best Model Updated with Chunk-Level F1-score: {best_f1_chunk:.2f}%")

    return train_losses, val_losses, f1_chunk, tag_accuracy
def plot_losses(train_losses, val_losses, title, model_name):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title)


    plt.savefig(f"{model_name}_loss_plot.png")
    plt.close()  

def preprocess_and_evaluate(model_path, test_file, vocab, embedding_tensor, model_type="RNN"):
   
    print("Preprocessing test data...")
    test_output = "test_preprocessed.json"  # Save preprocessed test data
    preprocessed_data(test_file, test_output)  #Preprocess test.json first
    
    print("Loading preprocessed test data...")
    test_data = load_data(test_output)
    
    print("Preparing test data...")
    test_loader = prepare_data(test_data, vocab)
    
    print(f"Loading trained model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Re-initialize model structure
    model = AspectTermExtractor(len(vocab), embedding_tensor.shape[1], 32, 3, embedding_tensor, model_type=model_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    print("Evaluating model on test data...")
    f1_chunk, tag_accuracy = calculate_f1_score(model, test_loader, vocab)

    print(f"Final Results for {model_type}:")
    print(f"Chunk-Level F1-Score: {f1_chunk:.2f}%")
    print(f"Tag-Level Accuracy: {tag_accuracy:.2f}%")

    return f1_chunk, tag_accuracy


if __name__=="__main__":

# TASK 1.1 - 1.3

  train_file = "train.json"
  val_file = "val.json"
  train_output = "train_task_1.json"
  test_output = "test_task_1.json"
  val_output = "val_task_1.json"

#   preprocessed_data(train_file,train_output)
#   preprocessed_data(val_file,val_output)
#   preprocessed_data(test_file,test_output)

### TASK 1.4
  print("Loading the data...")
  train_data = load_data("train_task_1.json")
  val_data = load_data("val_task_1.json")

  print("Building the vocabulary")
  vocabulary = build_vocabulary(train_data)
  
  glove_path = "glove.6B.50d.txt"
  fastText_path = "crawl-300d-2M.vec"

  glove_pickle_path = "glove_50d.pkl"  # Pickle file for fast loading
  fastText_pickle_path = "fastText_300d.pkl"

  glove_dim = 50 
  fastText_dim = 300

  gloveEmbeddings = load_embeddings(glove_path,glove_dim,glove_pickle_path)
  fastTexEmbeddings = load_embeddings(fastText_path,fastText_dim,fastText_pickle_path)

  
  print("Preparing data...")
  train_loader = prepare_data(train_data,vocabulary)
  val_loader = prepare_data(val_data,vocabulary)


  # Convert vocabulary to a list
  vocab_list = list(vocabulary.keys())  # Get all words from vocabulary

  #Convert GloVe embeddings to tensor
  glove_matrix = np.array([gloveEmbeddings.get(w, np.zeros(glove_dim, dtype=np.float32)) for w in vocab_list])
  glove_tensor = torch.tensor(glove_matrix, dtype=torch.float32)

  #Convert fastText embeddings to tensor
  fasttext_matrix = np.array([fastTexEmbeddings.get(w, np.zeros(fastText_dim, dtype=np.float32)) for w in vocab_list])
  fasttext_tensor = torch.tensor(fasttext_matrix, dtype=torch.float32)

  # Training RNN + GloVe...
print("Training RNN + GloVe...")
rnn_glove = AspectTermExtractor(len(vocabulary), 50, 64, 3, glove_tensor, model_type="RNN")
train_losses_rnn_glove, val_losses_rnn_glove, f1_rnn_glove, acc_rnn_glove = train_model(
    rnn_glove, train_loader, val_loader, vocabulary, "rnn_glove_best.pth"
)

# Training RNN + fastText...
print("Training RNN + fastText...")
rnn_fasttext = AspectTermExtractor(len(vocabulary), 300, 32, 3, fasttext_tensor, model_type="RNN")
train_losses_rnn_fasttext, val_losses_rnn_fasttext, f1_rnn_fasttext, acc_rnn_fasttext = train_model(
    rnn_fasttext, train_loader, val_loader, vocabulary, "rnn_fasttext_best.pth"
)

# Training GRU + GloVe...
print("Training GRU + GloVe...")
gru_glove = AspectTermExtractor(len(vocabulary), 50, 64, 3, glove_tensor, model_type="GRU")
train_losses_gru_glove, val_losses_gru_glove, f1_gru_glove, acc_gru_glove = train_model(
    gru_glove, train_loader, val_loader, vocabulary, "gru_glove_best.pth"
)

# Training GRU + fastText...
print("Training GRU + fastText...")
gru_fasttext = AspectTermExtractor(len(vocabulary), 300, 64, 3, fasttext_tensor, model_type="GRU")
train_losses_gru_fasttext, val_losses_gru_fasttext, f1_gru_fasttext, acc_gru_fasttext = train_model(
    gru_fasttext, train_loader, val_loader, vocabulary, "gru_fasttext_best.pth"
)

# Print the final evaluation scores after all models have been trained
print(f"RNN + GloVe - Chunk-Level F1: {f1_rnn_glove:.2f}% | Tag-Level Accuracy: {acc_rnn_glove:.2f}%")
print(f"RNN + fastText - Chunk-Level F1: {f1_rnn_fasttext:.2f}% | Tag-Level Accuracy: {acc_rnn_fasttext:.2f}%")
print(f"GRU + GloVe - Chunk-Level F1: {f1_gru_glove:.2f}% | Tag-Level Accuracy: {acc_gru_glove:.2f}%")
print(f"GRU + fastText - Chunk-Level F1: {f1_gru_fasttext:.2f}% | Tag-Level Accuracy: {acc_gru_fasttext:.2f}%")

# Plotting training and validation losses after all models have been trained
print("Plotting training losses...")
plot_losses(train_losses_rnn_glove, val_losses_rnn_glove, "RNN + GloVe Loss", "rnn_glove")
plot_losses(train_losses_rnn_fasttext, val_losses_rnn_fasttext, "RNN + fastText Loss", "rnn_fasttext")
plot_losses(train_losses_gru_glove, val_losses_gru_glove, "GRU + GloVe Loss", "gru_glove")
plot_losses(train_losses_gru_fasttext, val_losses_gru_fasttext, "GRU + fastText Loss", "gru_fasttext")


# TESTING PART
model_path = "gru_fasttext_best.pth"  
embedding_tensor = fasttext_tensor  
model_type = "GRU"  

test_file = "test.json"  

print(f"Evaluating {model_type} with chosen embeddings...")
f1_score, accuracy = preprocess_and_evaluate(model_path, test_file, vocabulary, embedding_tensor, model_type)

# Print the final results
print(f"Final Results for {model_type}:")
print(f"Chunk-Level F1-Score: {f1_score:.2f}%")
print(f"Tag-Level Accuracy: {accuracy:.2f}%")
