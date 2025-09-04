import torch.nn as nn

class ConvLSTM_Encoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, seq_len, dropout):
        """
        ConvLSTM Encoder module.
        
        :param in_size: Size of the input features.
        :param hidden_size: Number of features in the hidden state.
        :param out_size: Size of the output feature vector.
        :param num_layers: Number of stacked LSTM layers.
        :param seq_len: Length of the input sequence.
        :param dropout: Dropout rate for regularization.
        """
        super(ConvLSTM_Encoder, self).__init__()

        lstm_dropout = 0 if num_layers == 1 else dropout
        
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.conv = nn.Conv1d(hidden_size, out_size, kernel_size=seq_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for ConvLSTM Encoder.

        :param x: Input tensor of shape (batch_size, seq_len, in_size).
        :return: Encoded output tensor of shape (batch_size, out_size).
        """
        x, _ = self.lstm(x)

        x = self.dropout(x)

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.squeeze(2)
        
        return x

class ConvLSTM_Decoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, seq_len, dropout):
        """
        ConvLSTM Decoder module.
        
        :param in_size: Size of the input features.
        :param hidden_size: Number of features in the hidden state.
        :param out_size: Size of the output feature vector.
        :param num_layers: Number of stacked LSTM layers.
        :param seq_len: Length of the output sequence.
        :param dropout: Dropout rate for regularization.
        """
        super(ConvLSTM_Decoder, self).__init__()

        lstm_dropout = 0 if num_layers == 1 else dropout
        
        self.lstm = nn.LSTM(hidden_size, in_size, num_layers, batch_first=True, dropout=lstm_dropout)
        self.conv_transpose = nn.ConvTranspose1d(in_channels=out_size, out_channels=hidden_size, kernel_size=seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass for the ConvLSTM Decoder.
        
        :param x: Encoded input tensor of shape (batch_size, out_size).
        :return: Decoded output tensor of shape (batch_size, seq_len, in_size).
        """
        x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.conv_transpose(x)
        x = x.transpose(1, 2)

        x = self.dropout(x)

        x, _ = self.lstm(x)
        
        return x

class ConvLSTM_Autoencoder(nn.Module):
    def __init__(self, seq_len, num_feats, latent_seq_len, latent_num_feats, hidden_size, num_layers, dropout=0.5):
        """
        ConvLSTM-based Autoencoder module combining an encoder and a decoder. This module integrates 
        convolutional layers with LSTMs to effectively process spatiotemporal data and uses 1D 
        convolutional layers to reduce the sequence length.
        
        :param seq_len: Length of the input sequence.
        :param num_feats: Number of features in the input.
        :param latent_seq_len: Length of the latent sequence.
        :param latent_num_feats: Number of features in the latent representation.
        :param hidden_size: Number of hidden units in the LSTM.
        :param num_layers: Number of LSTM layers.
        :param dropout: Dropout rate for regularization.
        """
        super(ConvLSTM_Autoencoder, self).__init__()

        self.latent_seq_len = latent_seq_len
        self.latent_num_feats = latent_num_feats
        
        self.encoder = ConvLSTM_Encoder(in_size=num_feats, 
                                        hidden_size=hidden_size, 
                                        out_size=latent_seq_len * latent_num_feats, 
                                        num_layers=num_layers,
                                        seq_len=seq_len,
                                        dropout=dropout)

        self.decoder = ConvLSTM_Decoder(in_size=num_feats, 
                                        hidden_size=hidden_size, 
                                        out_size=latent_seq_len * latent_num_feats, 
                                        num_layers=num_layers,
                                        seq_len=seq_len,
                                        dropout=dropout)                       
    
    def forward(self, x):
        """
        Forward pass for the ConvLSTM Autoencoder.
        
        :param x: Input tensor of shape (batch_size, seq_len, num_feats).
        :return: Decoded output and latent representation.
        """
        enc_x = self.encoder(x)
        dec_x = self.decoder(enc_x)

        latent = enc_x.view(enc_x.size(0), self.latent_seq_len, self.latent_num_feats)
        
        return dec_x, latent