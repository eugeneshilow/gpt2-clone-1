from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_heads = config.n_heads
        self.n_embed = config.n_embed
        self.register_buffer("bias", torch.tril(torch.ones_like(torch.zeros(config.block_size, config.block_size))))
        
    def forward(self, x):
        
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)        
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y
        

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_blocks: int = 12
    n_heads: int = 12
    n_embed: int = 768


class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_blocks)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
    @classmethod
    def from_pretrained(cls, model_type: str):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        
        config_args = {
            'gpt2':         dict(n_blocks=12, n_heads=12, n_embed=768),
            'gpt2-medium':  dict(n_blocks=24, n_heads=16, n_embed=1024),
            'gpt2-large':   dict(n_blocks=36, n_heads=20, n_embed=1280),
            'gpt2-xl':      dict(n_blocks=48, n_heads=25, n_embed=1600),
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith("bias")]
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        hf_to_my_keys = {
            'transformer.wte.weight': 'wte.weight',
            'transformer.wpe.weight': 'wpe.weight',
            'transformer.ln_f.weight': 'ln_f.weight',
            'transformer.ln_f.bias': 'ln_f.bias',
            'lm_head.weight': 'lm_head.weight',
        }
        
        for k in sd_keys_hf:
            new_k = k.replace('transformer.h.', 'h.')
            new_k = hf_to_my_keys.get(k, new_k)
            
            if new_k in sd:
                if any(k.endswith(w) for w in transposed):
                    with torch.no_grad():
                        sd[new_k].copy_(sd_hf[k].t())
                else:
                    with torch.no_grad():
                        sd[new_k].copy_(sd_hf[k])
        
        return model
        
# -----------------------------------------------------------------------------

model = GPT.from_pretrained("gpt2")
print("didn't crash!")

