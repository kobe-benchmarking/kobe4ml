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

class Classifier(nn.Module):
    def __init__(self, in_size=3, out_size=5, num_heads=1, dropout=0.5, num_layers=1):
        """
        Classifier model for classification, combining an embedding layer with multi-head attention 
        and a classifier. The embedding captures complex dependencies in the input data, while the 
        classifier produces logits for classification.

        :param in_size: Size of the input features.
        :param out_size: Size of the output classes.
        :param num_heads: Number of attention heads.
        :param dropout: Dropout rate for regularization.
        :param num_layers: Number of attention layers.
        """
        super(Classifier, self).__init__()
        
        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(d_model=in_size, num_heads=num_heads) for _ in range(num_layers)
        ])

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_size, out_size)

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights and biases of the classifier linear layer:
        - Set the bias of the classifier linear layer to zero.
        - Initialize the weights with values drawn from a Xavier uniform distribution.
        """
        self.linear.bias.data.zero_()
        nn.init.xavier_uniform_(self.linear.weight.data)             
    
    def forward(self, x):
        """
        Forward pass for Classifier.
        
        :param x: Input tensor of shape (batch_size, seq_len, num_feats).
        :return: Tuple containing:
            - Logits of shape (batch_size, seq_len, out_size) for classification.
            - Attention matrix tensor of shape (batch_size, seq_len, in_size).
        """
        for attn_layer in self.attn_layers:
            output, attn_matrix = attn_layer(Q=x, K=x, V=x)
            x = x + output
        
        x = self.dropout(x)
        x = self.relu(x)

        logits = self.linear(x)
        
        return logits, attn_matrix