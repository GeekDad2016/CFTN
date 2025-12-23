import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TextEmbedder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
    def forward(self, x):
        b, t = x.shape
        return self.tok_emb(x) + self.pos_emb[:, :t, :]

class CallosalLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
        self.gate = nn.Parameter(torch.tensor(0.0)) 
    def forward(self, x, x_contralateral, self_attn_mask=None):
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=self_attn_mask)
        x = x + attn_out
        x_norm = self.norm_cross(x)
        other_norm = self.norm_cross(x_contralateral)
        cross_out, _ = self.cross_attn(query=x_norm, key=other_norm, value=other_norm)
        x = x + (torch.sigmoid(self.gate) * cross_out)
        x = x + self.ff(self.norm2(x))
        return x

class StandardLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
    def forward(self, x, attn_mask=None):
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x

class BiHemisphericBrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.get('n_embd', 256)
        self.nhead = config.get('n_head', 8)
        self.total_layers = config.get('n_layer', 12)
        self.max_seq_len = config.get('text_block_size', 128)
        self.vis_seq_len = config.get('block_size', 1024)
        self.text_vocab_size = config.get('text_vocab_size', 30522)
        self.vis_vocab_size = config.get('vocab_size', 4096)
        self.bridge_ratio = config.get('bridge_ratio', 0.33)
        
        bridge_len = int(self.total_layers * self.bridge_ratio)
        self.bridge_start = (self.total_layers - bridge_len) // 2
        self.bridge_end = self.bridge_start + bridge_len

        self.text_embed = TextEmbedder(self.text_vocab_size, self.d_model, self.max_seq_len)
        self.text_head = nn.Linear(self.d_model, self.text_vocab_size)
        
        self.vision_embed = nn.Embedding(self.vis_vocab_size + 1, self.d_model)
        self.vision_pos = nn.Parameter(torch.randn(1, self.vis_seq_len, self.d_model) * 0.02)
        self.vision_head = nn.Linear(self.d_model, self.vis_vocab_size)

        self.blank_text_emb = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.blank_vis_emb = nn.Parameter(torch.randn(1, self.vis_seq_len, self.d_model))

        self.layers_L = nn.ModuleList()
        self.layers_R = nn.ModuleList()
        for i in range(self.total_layers):
            is_bridge = self.bridge_start <= i < self.bridge_end
            if is_bridge:
                self.layers_L.append(CallosalLayer(self.d_model, self.nhead))
                self.layers_R.append(CallosalLayer(self.d_model, self.nhead))
            else:
                self.layers_L.append(StandardLayer(self.d_model, self.nhead))
                self.layers_R.append(StandardLayer(self.d_model, self.nhead))

    def forward(self, img_indices=None, text_ids=None):
        batch_size = img_indices.shape[0] if img_indices is not None else text_ids.shape[0]

        # Vision Path (Right Hemisphere)
        if img_indices is not None:
            h_R = self.vision_embed(img_indices) + self.vision_pos
        else:
            h_R = self.blank_vis_emb.expand(batch_size, -1, -1)

        # Text Path (Left Hemisphere)
        if text_ids is not None:
            h_L = self.text_embed(text_ids)
            seq_len = h_L.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(h_L.device)
        else:
            h_L = self.blank_text_emb.expand(batch_size, self.max_seq_len, -1)
            causal_mask = None

        for i, (lL, lR) in enumerate(zip(self.layers_L, self.layers_R)):
            is_bridge = self.bridge_start <= i < self.bridge_end
            if is_bridge:
                prev_L, prev_R = h_L, h_R
                h_L = lL(prev_L, x_contralateral=prev_R, self_attn_mask=causal_mask)
                h_R = lR(prev_R, x_contralateral=prev_L)
            else:
                h_L = lL(h_L, attn_mask=causal_mask)
                h_R = lR(h_R)
        
        return self.text_head(h_L), self.vision_head(h_R)
