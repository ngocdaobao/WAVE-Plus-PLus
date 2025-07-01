import torch
import torch.nn as nn
import torch.nn.functional as F

class Prompt(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.length = args.prompt_length
        self.embed_dim = args.prompt_embed_dim
        self.pool_size = args.prompt_pool_size
        self.top_k = args.prompt_top_k
        self.prompt_type = args.prompt_type
        
        # total prompt length
        self.total_prompt_length = self.top_k * self.length

        # prompt
        prompt_pool_shape = (self.pool_size, self.length, self.embed_dim)
        if args.prompt_init == "zero":
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif args.prompt_init == "uniform":
            self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
            nn.init.uniform_(self.prompt, -1, 1)
        else:
            raise Exception("Not support type of prompt initialization")

        # prompt_key
        key_shape = (self.pool_size, self.embed_dim * 2)
        if args.prompt_key_init == "zero":
            self.prompt_key = nn.Parameter(torch.zeros(key_shape))
        elif args.prompt_key_init == "uniform":
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            raise Exception("Not support type of prompt key initialization")                                        


    def forward(self, x_embed, x_key=None):
        out = dict()
        
        prompt_key_norm = nn.functional.normalize(self.prompt_key, dim=1)
        x_key_norm = nn.functional.normalize(x_key, dim=1)

        similarity = torch.matmul(x_key_norm, prompt_key_norm.t())
        softmax_sim = F.softmax(similarity, dim=1)
        _, _id = torch.topk(similarity, k=self.top_k, dim=1)
        
        reshaped_prompt = self.prompt.view(self.pool_size, -1)
        softmax_sim = softmax_sim.unsqueeze(2).expand(x_embed.shape[0], self.pool_size, reshaped_prompt.shape[-1]) 

        result = softmax_sim * reshaped_prompt.unsqueeze(0) 
        mean_result = torch.mean(result, dim=1)
        
        x_key_norm = x_key_norm.unsqueeze(1)
        out["reduce_sim"] = torch.sum(prompt_key_norm[_id] * x_key_norm) / x_embed.shape[0]
        mean_result_reshaped = mean_result.view(x_embed.shape[0], self.length, self.embed_dim)
        out["prompted_embedding"] = torch.cat([mean_result_reshaped, x_embed], dim=1)
    
        return out
