# -*- coding: utf-8 -*-
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 设备检查和选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "data"
corpus_file = os.path.join(data_dir, "侠客行.txt")

# 加载语料
def load_corpus(file_path):
    encodings = ['utf-8', 'gb18030', 'gbk']
    for encoding in encodings:
        try:
            with open(file_path, encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Failed to read {file_path} with available encodings")

corpus = load_corpus(corpus_file)

# 使用字符和标点作为基本单位
def preprocess(text):
    return list(text)

tokens = preprocess(corpus)
vocab = sorted(set(tokens))

# 打印一些预处理后的数据
print(f"Vocabulary size: {len(vocab)}")
print(f"Sample tokens: {tokens[:10]}")

# 构建词汇表
token_index = {token: idx for idx, token in enumerate(vocab, start=1)}
token_index["<UNK>"] = 0  # 未知字符处理

# 将文本转换为序列，低频词用<UNK>表示
sequences = [token_index.get(token, 0) for token in tokens]

# 生成训练数据
class TextDataset(Dataset):
    def __init__(self, sequences, seq_len=100):
        self.sequences = sequences
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences) - self.seq_len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx:idx + self.seq_len], dtype=torch.long),
            torch.tensor(self.sequences[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        )

dataset = TextDataset(sequences)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 定义Transformer模型
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, attn_mask=mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.mha2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        attn1, _ = self.mha1(x, x, x, attn_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)
        attn2, _ = self.mha2(out1, enc_output, enc_output, attn_mask=padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        return self.layernorm3(out2 + ffn_output)

class TransformerModel(nn.Module):
    def __init__(self, num_layers, num_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, num_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, num_model)
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(num_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(num_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.final_layer = nn.Linear(num_model, target_vocab_size)

    def forward(self, inp, tar, training=False):
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_output = self.encoder_embedding(inp)
        enc_output = enc_output.permute(1, 0, 2)  # Shape to (S, N, E) for PyTorch MultiheadAttention
        for i in range(len(self.encoder_layers)):
            enc_output = self.encoder_layers[i](enc_output, None)
        dec_output = self.decoder_embedding(tar)
        dec_output = dec_output.permute(1, 0, 2)
        for i in range(len(self.decoder_layers)):
            dec_output = self.decoder_layers[i](dec_output, enc_output, look_ahead_mask, None)
        dec_output = dec_output.permute(1, 0, 2)
        final_output = self.final_layer(dec_output)
        return final_output

    def create_masks(self, inp, tar):
        enc_padding_mask = self.create_padding_mask(inp).to(inp.device)
        dec_padding_mask = self.create_padding_mask(inp).to(inp.device)
        look_ahead_mask = self.create_look_ahead_mask(tar.size(1)).to(tar.device)
        dec_target_padding_mask = self.create_padding_mask(tar).to(tar.device)
        combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, look_ahead_mask, dec_padding_mask

    def create_padding_mask(self, seq):
        return (seq == 0).unsqueeze(1).unsqueeze(2)

    def create_look_ahead_mask(self, size):
        mask = torch.triu(torch.ones((size, size)), diagonal=1)
        return mask == 1

num_layers = 8
num_model = 128
dff = 256
num_heads = 8
input_vocab_size = len(vocab) + 2
target_vocab_size = len(vocab) + 2
dropout_rate = 0.1

model = TransformerModel(num_layers, num_model, num_heads, dff, input_vocab_size, target_vocab_size, 100, 100, dropout_rate)
model = model.to(device)
try:
    model.load_state_dict(torch.load('transformer_model_test.pth'))
    print('Model loaded successfully.')
except FileNotFoundError:
    print('Model not found, training from scratch.')

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
model.train()
for epoch in range(50):
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.view(-1, target_vocab_size), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

    # 打印一些样本输出
    src_sample = src[0].cpu().numpy()
    tgt_sample = tgt[0].cpu().numpy()
    output_sample = torch.argmax(output[0], dim=1).cpu().numpy()
    print(f"Source: {''.join([vocab[i - 1] for i in src_sample if i > 0])}")
    print(f"Target: {''.join([vocab[i - 1] for i in tgt_sample if i > 0])}")
    print(f"Output: {''.join([vocab[i - 1] for i in output_sample if i > 0])}")

    # 保存模型
    torch.save(model.state_dict(), 'transformer_model_test.pth')

# 文本生成函数
def generate_text(model, start_string, num_generate=100):
    input_eval = [token_index.get(s, 0) for s in start_string]
    input_eval = torch.tensor(input_eval, dtype=torch.long).unsqueeze(0).to(device)
    text_generated = []
    decoder_input = torch.tensor([[token_index.get('。', 0)]], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(num_generate):
            predictions = model(input_eval, decoder_input)
            predictions = predictions[:, -1, :]
            predicted_id = torch.multinomial(torch.softmax(predictions, dim=-1), num_samples=1).item()
            input_eval = torch.cat([input_eval, torch.tensor([[predicted_id]], dtype=torch.long).to(device)], dim=-1)
            decoder_input = torch.cat([decoder_input, torch.tensor([[predicted_id]], dtype=torch.long).to(device)], dim=-1)
            text_generated.append(vocab[predicted_id - 1] if predicted_id > 0 else '<UNK>')

    return start_string + ''.join(text_generated)

print(generate_text(model, start_string="田青文接过羽箭，只看了一眼，"))
