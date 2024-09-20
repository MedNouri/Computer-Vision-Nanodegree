import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Using ResNet-34 for a lighter model
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # TODO: Complete this function
       
        # Set the size of the embedding, hidden, and output layers
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Define the embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Define the LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        torch.nn.init.kaiming_normal_(self.linear.weight)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])  # Exclude the <end> token
        # TODO: Complete this function

        # Concatenate the image features and caption embeddings
        embedding = torch.cat((features.unsqueeze(dim = 1), embeddings), dim = 1)

        # Pass the embeddings through the LSTM layer
        lstm_out, hidden = self.lstm(embedding)

        # Pass the LSTM output through the output layer
        outputs = self.linear(lstm_out)
        return outputs
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_sentence = []
        for i in range(max_len):
            # Pass the inputs and hidden states through the LSTM layer
            hiddens, states = self.lstm(inputs, states)
            
            # Pass the LSTM output through the linear layer to get the predicted word scores
            outputs = self.linear(hiddens.squeeze(1))
           
           # Get the index of the word with the highest score
            _, predicted = outputs.max(1)
            predicted_sentence.append(predicted.item())
            
            # Embed the predicted word index to get the input for the next timestep
            inputs = self.embed(predicted).unsqueeze(1)
        return predicted_sentence