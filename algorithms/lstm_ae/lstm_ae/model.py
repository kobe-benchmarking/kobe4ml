import torch.nn as nn

class LSTM_Encoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout):
        """
        LSTM Encoder module.
        
        :param in_size: Size of the input features.
        :param hidden_size: Number of features in the hidden state.
        :param out_size: Size of the output feature vector.
        :param num_layers: Number of stacked LSTM layers.
        :param dropout: Dropout rate for regularization.
        """
        super(LSTM_Encoder, self).__init__()

        lstm_dropout = 0 if num_layers == 1 else dropout
        
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.fc = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for LSTM Encoder.
        
        :param x: Input tensor of shape (batch_size, seq_len, in_size).
        :return: Encoded output tensor of shape (batch_size, out_size).
        """
        _, seq_len, _ = x.size()

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = self.dropout(x)
        x = self.fc(x)
        
        return x, seq_len

class LSTM_Decoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout):
        """
        LSTM Decoder module.
        
        :param in_size: Size of the input features.
        :param hidden_size: Number of features in the hidden state.
        :param out_size: Size of the output feature vector.
        :param num_layers: Number of stacked LSTM layers.
        :param dropout: Dropout rate for regularization.
        """
        super(LSTM_Decoder, self).__init__()

        lstm_dropout = 0 if num_layers == 1 else dropout
        
        self.fc = nn.Linear(out_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, in_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, seq_len):
        """
        Forward pass for LSTM Decoder.
        
        :param x: Encoded input tensor of shape (batch_size, out_size).
        :return: Decoded output tensor of shape (batch_size, seq_len, in_size).
        """
        x = self.fc(x)
        x = self.dropout(x)
        x = x.unsqueeze(1).repeat(1, seq_len, 1)

        x, _ = self.lstm(x)
        
        return x

class LSTM_Autoencoder(nn.Module):
    def __init__(self, num_feats, latent_seq_len, latent_num_feats, hidden_size, num_layers, dropout=0.5):
        """
        LSTM-based Autoencoder module combining an encoder and a decoder. This module uses LSTM layers 
        to capture temporal dependencies in sequential data and reduces the sequence length by taking 
        the last element of the sequence.
        
        :param num_feats: Number of features in the input.
        :param latent_seq_len: Length of the latent sequence.
        :param latent_num_feats: Number of features in the latent representation.
        :param hidden_size: Number of hidden units in the LSTM.
        :param num_layers: Number of LSTM layers.
        :param dropout: Dropout rate for regularization.
        """
        super(LSTM_Autoencoder, self).__init__()

        self.latent_seq_len = latent_seq_len
        self.latent_num_feats = latent_num_feats
        
        self.encoder = LSTM_Encoder(in_size=num_feats, 
                                    hidden_size=hidden_size, 
                                    out_size=latent_seq_len * latent_num_feats, 
                                    num_layers=num_layers,
                                    dropout=dropout)

        self.decoder = LSTM_Decoder(in_size=num_feats, 
                                    hidden_size=hidden_size, 
                                    out_size=latent_seq_len * latent_num_feats, 
                                    num_layers=num_layers,
                                    dropout=dropout)                       
    
    def forward(self, x):
        """
        Forward pass for LSTM Autoencoder.
        
        :param x: Input tensor of shape (batch_size, seq_len, num_feats).
        :return: Decoded output and latent representation.
        """
        enc_x, seq_len = self.encoder(x)
        dec_x = self.decoder(enc_x, seq_len)

        latent = enc_x.view(enc_x.size(0), self.latent_seq_len, self.latent_num_feats)
        
        return dec_x, latent