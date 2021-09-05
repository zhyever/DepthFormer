import torch
import torch.nn as nn

a = nn.Embedding(300, 64*2)
query_embed, tgt = torch.split(a.weight, 64, dim=1)
print(tgt.shape)
b = tgt.unsqueeze(0).expand(8, -1, -1)

print(b.shape)