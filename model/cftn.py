import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BidirectionalAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, is_cross_attn=False):
        super().__init__()
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.is_cross_attn = is_cross_attn

    def forward(self, x, context=None):
        B, T, C = x.size()
        # For cross-attention, K and V come from text; Q comes from image
        k_source = context if self.is_cross_attn else x
        v_source = context if self.is_cross_attn else x
        
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(k_source).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(v_source).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class CFTNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.self_attn = BidirectionalAttention(config['n_embd'], config['n_head'], config['attn_pdrop'], config['resid_pdrop'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.cross_attn = BidirectionalAttention(config['n_embd'], config['n_head'], config['attn_pdrop'], config['resid_pdrop'], is_cross_attn=True)
        self.ln3 = nn.LayerNorm(config['n_embd'])
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.GELU(),
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(config['resid_pdrop']),
        )

    def forward(self, x, text_features):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), context=text_features)
        x = x + self.mlp(self.ln3(x))
        return x

class CFTN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config['block_size'] # 1024
        
        # Image Tokenizer
        self.tok_emb = nn.Embedding(config['vocab_size'] + 1, config['n_embd']) # +1 for [MASK]
        self.pos_emb = nn.Parameter(torch.zeros(1, config['block_size'], config['n_embd']))
        
        # Text Tower (Simple Embedding + Projection for now)
        self.text_tok_emb = nn.Embedding(config['text_vocab_size'], config['n_embd'])
        self.text_pos_emb = nn.Parameter(torch.zeros(1, config['text_block_size'], config['n_embd']))
        
        self.blocks = nn.ModuleList([CFTNBlock(config) for _ in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        self.mask_token_id = config['vocab_size']

    def forward(self, img_idx, text_idx):
        # Image Path
        x = self.tok_emb(img_idx) + self.pos_emb
        
        # Text Path
        text_f = self.text_tok_emb(text_idx) + self.text_pos_emb
        
        for block in self.blocks:
            x = block(x, text_f)
            
        return self.head(self.ln_f(x))