import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 设备检查和选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
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


# 直接使用字符和标点作为基本单位
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


# 定义Seq2Seq模型
class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        decoder_outputs, _ = self.decoder(tgt, (hidden, cell))
        output = self.fc(decoder_outputs)
        return output


embed_size = 512
hidden_size = 1024

# 加载模型
model = Seq2SeqModel(len(vocab) + 1, embed_size, hidden_size).to(device)
try:
    model.load_state_dict(torch.load('seq2seq_model_char.pth'))
    print('successfully load')
except:
    print('model not exist')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
model.train()
for epoch in range(50):
    print(f'Epoch {epoch + 1} start')
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, len(vocab) + 1), tgt[:, 1:].reshape(-1))
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
    torch.save(model.state_dict(), 'seq2seq_model_char.pth')
