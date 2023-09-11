import torch
import torch.nn as nn
import torch.nn.functional as F
from prettytable import PrettyTable
import math
from tqdm import tqdm
from einops import rearrange, repeat
import time

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


#only enable for live visualization of results
#import mcpi.minecraft as minecraft
#mc = minecraft.Minecraft.create()

#only enable for live visualization of results
#def place_voxel_blocks_in_minecraft(voxel,position):
#    
#    """
#    Place the blocks from the voxel in the Minecraft world starting from the given point.
#    
#    Args:
#    - start_x, start_y, start_z: Starting coordinates.
#    - voxel: 3D array of shape 8x8x8 containing block types (and possibly subtypes).
    
#    """
#    start_x,start_y,start_z = position
#    for z in range(8):
#        for y in range(8):
#            for x in range(8):
#                block_info = voxel[y, x, z]
#                mc.setBlock(start_x + x, start_y + y, start_z + z, int(block_info))
                




def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class Block(nn.Module):
    def __init__(self,dim,n_heads,hidden_fc_dim=2048):
        super(Block, self).__init__()
        self.l_norm1 = nn.LayerNorm(dim)
        self.l_norm2 = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(dim,n_heads,batch_first=True,dropout=0.)

        self.fc = nn.Sequential(
            nn.Linear(dim,int(hidden_fc_dim*2)),
            SwiGLU(),
            nn.Linear(hidden_fc_dim,dim)
            )

    def forward(self,x):
        x_norm = self.l_norm1(x)
        
        attention_out,_ = self.attention(x_norm,x_norm,x_norm)
        x = x + attention_out

        x_norm = self.l_norm2(x)
        fw = self.fc(x_norm)
        x = x + fw
        return x

class GPT(nn.Module):
    def __init__(self,max_len,vocab_size,dim=768,heads=16,num_decoder=28):
        super(GPT, self).__init__()
        self.max_len = max_len
        self.mask_id = vocab_size

        self.embed = nn.Embedding(vocab_size+1,dim)
        self.positional_embedding = nn.Embedding(self.max_len,dim)

        self.decoder = nn.Sequential(
            *[Block(dim,heads,hidden_fc_dim=dim*4)for _ in range(num_decoder)]
            )

        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim,vocab_size,bias=False)

    def forward(self,x,target=None):
        bs,sl = x.shape

        if target!=None:
            rand_time = torch.zeros((bs,1), device = x.device).float().uniform_(0, 1)
            rand_mask_probs = torch.cos(rand_time * math.pi * 0.5)
            num_token_masked = (sl * rand_mask_probs).round().clamp(min = 1)

            batch_randperm = torch.rand((bs, sl), device = x.device).argsort(dim = -1)
            mask = batch_randperm < num_token_masked

            x = torch.where(mask,self.mask_id,x)
            
        
        x = self.embed(x)

        positions = (torch.arange(0, self.max_len).unsqueeze(0).expand(x.shape[0], -1)).cuda()
        x = x + self.positional_embedding(positions)

        x = self.decoder(x)

        x = self.ln(x)
        logits = self.fc(x)

        if target!=None:
            logits = logits.reshape(-1,logits.shape[2])
        
            #target = torch.where(mask,target,torch.tensor(-1).to(target.device))
            target = target.reshape(-1)
            loss = F.cross_entropy(logits,target,ignore_index=-1)
            return loss
        return logits
    

    def generate(self, tokens, timesteps=12, random_steps=8, temperature=1, topk_filter_thres=.5):
        bs, sl = tokens.shape
        ids = tokens.cuda()
        starting_temperature = temperature
        random=True
        if random_steps==0:
            random= False

        #only enable for live visualization of results
        #player_pos = mc.player.getPos()

        actual_masked = [int(torch.cos(t * math.pi * 0.5) * sl) for t in torch.linspace(0, 1, timesteps, device=tokens.device)]
        for i in tqdm(range(timesteps)):
            if i < timesteps - 1:
                num_tokens_to_unmask = actual_masked[i] - actual_masked[i + 1]
            else:
                num_tokens_to_unmask = actual_masked[i]
            num_tokens_to_unmask = max(num_tokens_to_unmask, 1)
            #print(ids,"IDS")

            logits = self.forward(ids)
            mask = ids == self.mask_id        
            logits = torch.where(mask[:, :, None], logits, 0)


            temperature = starting_temperature * ((timesteps-i)/timesteps)
            
            filtered_logits = top_k(logits, topk_filter_thres)
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            pred_ids = torch.where(mask, pred_ids, ids)

            #print(pred_ids,"PRED_IDS")
            if random:
                masked_indices = torch.nonzero(mask[0]).squeeze()
                random_indices = torch.randperm(masked_indices.size(0))[:num_tokens_to_unmask]
                sample_index = masked_indices[random_indices]
            else:
                probs_without_temperature = logits.softmax(dim=-1)
                probs_without_temperature = probs_without_temperature.gather(2, pred_ids[..., None])[:, :, 0]
                _, sample_index = torch.topk(probs_without_temperature[0], num_tokens_to_unmask)
            #print(sample_index,"SAMPLE_IDX")

            for index in sample_index:
                ids[:, index] = pred_ids[:, index]
            if i > random_steps:
                random= False

            #only enable for live visualization of results
            #place_voxel_blocks_in_minecraft(torch.where(ids==831,torch.zeros_like((ids)),ids).view((8,8,8)),player_pos)
            if not (ids == self.mask_id).any():
                return ids
            #time.sleep(.2)
    
   

if __name__ == '__main__':
    gpt = GPT(512,831,256,16,1).cuda()
    count_parameters(gpt)

    x = torch.zeros(1,512).long().cuda()
    out = gpt(x, target=x)
    print(out.shape)
