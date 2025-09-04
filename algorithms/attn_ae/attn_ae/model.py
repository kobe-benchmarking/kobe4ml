import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Initialize the Multi-Head Attention module. The module computes the attention matrix 
        with shape torch.Size([batch_size, seq_len, d_model]), which can be accessed 
        as model.encoder[layer_id].self_attn.attn_matrix.

        :param d_model: dimension of the input and output features
        :param num_heads: number of attention heads
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_matrix = None
        
    def attention_scores(self, Q, K, V):
        """
        Calculate the attention scores and apply them to the values.
        
        :param Q: Query matrix.
        :param K: Key matrix.
        :param V: Value matrix.
        :return: Attention scores per head.
        """
        dot_product = torch.matmul(Q, K.transpose(-2, -1))

        scaling_factor = math.sqrt(self.d_k)
        attn_scores = dot_product / scaling_factor

        attn_probs = torch.softmax(attn_scores, dim=-1)
        self.attn_weights = attn_probs

        attn_scores = torch.matmul(attn_probs, V)

        return attn_scores
        
    def split_heads(self, x):
        """
        Split the input into multiple heads.

        :param x: Tensor (batch_size, seq_len, d_model).
        :return: Tensor (batch_size, num_heads, seq_len, d_k).
        """
        batch_size, seq_len, _ = x.size()

        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        x = x.transpose(1, 2)
        
        return x
        
    def combine_heads(self, x):
        """
        Combine multiple heads into a single tensor.

        :param x: Tensor (batch_size, num_heads, seq_len, d_k).
        :return: Tensor (batch_size, seq_len, d_model).
        """
        batch_size, _, seq_len, _ = x.size()

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_len, self.d_model)
        
        return x
        
    def forward(self, Q, K, V):
        """
        Forward pass for multi-head attention.

        :param Q: Query matrix.
        :param K: Key matrix.
        :param V: Value matrix.
        :return: Multi-head attention matrix.
        """
        Q = self.split_heads(x=self.W_q(Q))
        K = self.split_heads(x=self.W_k(K))
        V = self.split_heads(x=self.W_v(V))
        
        attn_scores = self.attention_scores(Q, K, V)
        attn_matrix = self.combine_heads(attn_scores)
        
        output = self.W_o(attn_matrix)

        return output, attn_matrix

class Attn_Encoder(nn.Module):
    def __init__(self, in_size, out_size, num_heads, num_layers, seq_len, dropout):
        """
        Multi-Head Attention Encoder module.
        
        :param in_size: Size of the input features.
        :param out_size: Size of the output feature vector.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of attention layers.
        :param seq_len: Length of the input sequence.
        :param dropout: Dropout rate for regularization.
        """
        super(Attn_Encoder, self).__init__()
        
        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(d_model=in_size, num_heads=num_heads) for _ in range(num_layers)
        ])

        self.conv = nn.Conv1d(in_size, out_size, kernel_size=seq_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass for Attention Encoder.
        
        :param x: Input tensor of shape (batch_size, seq_len, in_size).
        :return: Encoded output tensor of shape (batch_size, out_size) and attention matrix of shape (batch_size, seq_len, in_size).
        """
        for attn_layer in self.attn_layers:
            output, attn_matrix = attn_layer(Q=x, K=x, V=x)
            x = x + output

        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.squeeze(2)
        
        return x, attn_matrix

class Attn_Decoder(nn.Module):
    def __init__(self, in_size, out_size, num_heads, num_layers, seq_len, dropout):
        """
        Multi-Head Attention Decoder module.
        
        :param in_size: Size of the input features.
        :param out_size: Size of the output feature vector.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of attention layers.
        :param seq_len: Length of the output sequence.
        :param dropout: Dropout rate for regularization.
        """
        super(Attn_Decoder, self).__init__()

        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(d_model=in_size, num_heads=num_heads) for _ in range(num_layers)
        ])
        
        self.conv_transpose = nn.ConvTranspose1d(in_channels=out_size, out_channels=in_size, kernel_size=seq_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass for Attention Decoder.
        
        :param x: Encoded input tensor of shape (batch_size, out_size).
        :return: Decoded output tensor and attention matrix, both of shape (batch_size, seq_len, in_size).
        """
        x = x.unsqueeze(1)
        x = x.transpose(1, 2)
        x = self.conv_transpose(x)
        x = x.transpose(1, 2)

        x = self.dropout(x)

        for attn_layer in self.attn_layers:
            output, attn_matrix = attn_layer(Q=x, K=x, V=x)
            x = x + output
        
        return x, attn_matrix

class Attn_Autoencoder(nn.Module):
    def __init__(self, seq_len, num_feats, latent_seq_len, latent_num_feats, num_heads, num_layers, dropout=0.5):
        """
        Multi-Head Attention-based Autoencoder module combining an encoder and a decoder. This module 
        uses multi-head attention mechanisms to capture complex dependencies in the input data, and 
        applies 1D convolutional layers to reduce the sequence length.
        
        :param seq_len: Length of the input sequence.
        :param num_feats: Number of features in the input.
        :param latent_seq_len: Length of the latent sequence.
        :param latent_num_feats: Number of features in the latent representation.
        :param num_heads: Number of attention heads.
        :param num_layers: Number of attention layers.
        :param dropout: Dropout rate for regularization.
        """
        super(Attn_Autoencoder, self).__init__()

        self.latent_seq_len = latent_seq_len
        self.latent_num_feats = latent_num_feats
        
        self.encoder = Attn_Encoder(in_size=num_feats,
                                    out_size=latent_seq_len * latent_num_feats,
                                    num_heads=num_heads,
                                    num_layers=num_layers,
                                    seq_len=seq_len,
                                    dropout=dropout)

        self.decoder = Attn_Decoder(in_size=num_feats,
                                    out_size=latent_seq_len * latent_num_feats,
                                    num_heads=num_heads,
                                    num_layers=num_layers,
                                    seq_len=seq_len,
                                    dropout=dropout)                       
    
    def forward(self, x):
        """
        Forward pass for Attention Autoencoder.
        
        :param x: Input tensor of shape (batch_size, seq_len, num_feats).
        :return: Decoded output, latent representation, and averaged attention matrix.
        """
        enc_x, enc_attn_matrix = self.encoder(x)
        dec_x, dec_attn_matrix = self.decoder(enc_x)

        latent = enc_x.view(enc_x.size(0), self.latent_seq_len, self.latent_num_feats)

        attn_matrix = (enc_attn_matrix + dec_attn_matrix) / 2

        return dec_x, latent, attn_matrix