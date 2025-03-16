# from methods.ct_loss import constractive_loss, sim
# import torch

# a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)

# b = torch.tensor([[[1, 2, 3], [1, 2, 1]], [[4, 5, 0], [4, 5, 3]], [[7, 8, 9], [1, 2, 3]]], dtype=torch.float32)

# c = torch.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]], dtype=torch.float32)

# print(sim(a, b))

# print(constractive_loss(a, b, c))

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

args = {
    "gpu": 0,
    "dataname": "TACRED",
    "task_name": "TACRED",
    "device": "cuda",
    "batch_size": 16,
    "num_tasks": 10,
    "rel_per_task": 8,
    "pattern": "entity_marker",
    "max_length": 256,
    "encoder_output_size": 768,
    "vocab_size": 30522,
    "marker_size": 4,
    "num_workers": 0,
    "classifier_lr": 1e-2,
    "encoder_lr": 1e-3,
    "prompt_pool_lr": 1e-3,
    "sgd_momentum": 0.1,
    "gmm_num_components": 1,
    "pull_constraint_coeff": 0.1,
    "classifier_epochs": 100,
    "encoder_epochs": 10,
    "prompt_pool_epochs": 10,
    "replay_s_e_e": 256,
    "replay_epochs": 100,
    "seed": 2021,
    "max_grad_norm": 10,
    "data_path": "datasets/",
    "bert_path": "bert-base-uncased",
    "cov_mat": True,
    "max_num_models": 10,
    "sample_freq": 20,
    "prompt_length": 1,
    "prompt_embed_dim": 768,
    "prompt_pool_size": 80,
    "prompt_top_k": 8,
    "prompt_init": "uniform",
    "prompt_key_init": "uniform",
    "prompt_type": "coda-prompt"
}

config = Config(**args)

from dataloaders.sampler import data_sampler

sampler = data_sampler(config)

for steps, (training_data, valid_data, test_data, current_relations, 
                    historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):

    print(len(training_data))
    print("="*20)
    print(len(seen_descriptions))