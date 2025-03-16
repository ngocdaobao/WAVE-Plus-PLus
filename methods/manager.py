from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader

from .swag import SWAG
from .model import *
from .backbone import *
from .prompt import *
from .utils import *
from .ct_loss import contrastive_loss, new_contrastive_loss
from collections import Counter

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from sklearn.mixture import GaussianMixture

from tqdm import tqdm, trange
import pickle
import wandb

import logging

logger = logging.getLogger(__name__)

class Manager(object):
    def __init__(self, args):
        super().__init__()
        
        # NgoDinhLuyen EoE
        self.expert_distribution = [
            {
                "class_mean": [],
                "accumulate_cov": torch.zeros(args.encoder_output_size * 2, args.encoder_output_size * 2),
                "cov_inv": torch.ones(args.encoder_output_size * 2, args.encoder_output_size * 2),
            }
        ]
        self.num_tasks = -1
        self.base_bert = BaseBert(config=args).to(args.device)
        self.query_mode = "mahalanobis"
        self.max_expert = -1
        self.eoeid2waveid = {}
        self.beta = args.beta
        # NgoDinhLuyen EoE

    def train_classifier(self, args, classifier, swag_classifier, replayed_epochs, name):
        classifier.train()
        swag_classifier.train()

        modules = [classifier]
        modules = nn.ModuleList(modules)
        modules_parameters = modules.parameters()

        optimizer = torch.optim.Adam([{"params": modules_parameters, "lr": args.classifier_lr}])

        def train_data(data_loader_, name=""):
            losses = []
            accuracies = []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            total_hits = 0
            for step, (labels, tokens, _) in enumerate(td):
                try:
                    optimizer.zero_grad()

                    # batching
                    sampled += len(labels)
                    targets = labels.type(torch.LongTensor).to(args.device)
                    tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)

                    # classifier forward
                    reps = classifier(tokens)

                    # prediction
                    probs = F.softmax(reps, dim=1)
                    _, pred = probs.max(1)
                    hits = (pred == targets).float()

                    # accuracy
                    total_hits += hits.sum().data.cpu().numpy().item()

                    # loss components
                    loss = F.cross_entropy(input=reps, target=targets, reduction="mean")
                    losses.append(loss.item())
                    loss.backward()
                    
                    # log to wandb
                    wandb.log({
                        "classifier_loss": loss.item(),
                        "classifier_acc": hits.mean().item()
                    })

                    # params update
                    torch.nn.utils.clip_grad_norm_(modules_parameters, args.max_grad_norm)
                    optimizer.step()

                    # display
                    td.set_postfix(loss=np.array(losses).mean(), acc=total_hits / sampled)
                except:
                    continue

        for e_id in range(args.classifier_epochs):
            data_loader = get_data_loader(args, replayed_epochs[e_id % args.replay_epochs], shuffle=True)
            train_data(data_loader, f"{name}{e_id + 1}")
            swag_classifier.collect_model(classifier)
            if e_id % args.sample_freq == 0 or e_id == args.classifier_epochs - 1:
                swag_classifier.sample(0.0)
                bn_update(data_loader, swag_classifier)

    # NgoDinhLuyen Add Function calculate negative period
    @torch.no_grad()
    def find_negative_labels(self, args, encoder, seen_description, k=4):
        negative_dict = dict()
        description_out = {}
        description_matrix = []
        
        rel2id = dict()
        with torch.no_grad():
            for idx, (rel, descriptions) in enumerate(seen_description.items()):
                rel2id[idx] = self.rel2id[rel]
                temp = []
                for description in descriptions:
                    des_tokens = torch.tensor([description['token_ids']]).to(args.device)
                    output = encoder(des_tokens, extract_type="cls")["cls_representation"]
                    output = output.detach().cpu()
                    temp.append(output)

                temp = torch.stack(temp, dim=0)
                temp = torch.mean(temp, dim=0)
                temp = temp.squeeze(0)
                description_out[self.rel2id[rel]] = temp
                description_matrix.append(temp)
            
            
        description_matrix = torch.stack(description_matrix, dim=0)
    
        # Tính cosine similarity giữa reps và descriptions
        similarities = sim(description_matrix, description_matrix) / 5  # (N, M)
        
        # Sắp xếp theo giá trị giảm dần (dim=1 để sắp theo hàng)
        _, topk_indices = torch.topk(similarities, k=min(k+1,description_matrix.shape[0]), dim=1)  # k+1 để bỏ chính nó
        
        # Bỏ chính nó (index đầu tiên)
        topk_indices = topk_indices[:, 1:].tolist()  # Chuyển thành list để dễ dùng
        
        for i in range(len(topk_indices)):
            new_topk_indices = [rel2id[j] for j in topk_indices[i]]
            negative_dict[rel2id[i]] = new_topk_indices
        return negative_dict
        
    def train_encoder(self, args, encoder, training_data, seen_description, task_id, beta=0.1):
        encoder.train()
        classifier = Classifier(args=args).to(args.device)
        classifier.train()
        data_loader = get_data_loader(args, training_data, shuffle=True)

        modules = [classifier, encoder.encoder.embeddings]
        modules = nn.ModuleList(modules)
        modules_parameters = modules.parameters()

        optimizer = torch.optim.Adam([{"params": modules_parameters, "lr": args.encoder_lr}])

        def train_data(data_loader_, name="", e_id=0):
            losses = []
            accuracies = []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            total_hits = 0 
            negative_dict = self.find_negative_labels(args, encoder, seen_description)
            for step, (labels, tokens, _) in enumerate(td):
                optimizer.zero_grad()

                # batching
                sampled += len(labels)
                targets = labels.type(torch.LongTensor).to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)

                # encoder forward
                encoder_out = encoder(tokens)
                
                if args.use_ct_in_encoder == "yes":             
                    # New
                    if args.type_ctloss == "new":
                        if step % 50 ==0:
                            negative_dict = self.find_negative_labels(args, encoder, seen_description)
                        
                        all_description_label_need_cal = []
                        
                        for label in labels:
                            label = int(label)
                            if label not in all_description_label_need_cal:
                                all_description_label_need_cal.append(label)
                            for lab in negative_dict[label]:
                                if lab not in all_description_label_need_cal:
                                    all_description_label_need_cal.append(lab)
                        
                        description_out = {}
                        for rel, descriptions in seen_description.items():
                            if self.rel2id[rel] in all_description_label_need_cal:
                                temp = []
                                for description in descriptions:
                                    des_tokens = torch.tensor([description['token_ids']]).to(args.device)
                                    temp.append(encoder(des_tokens, extract_type="cls")["cls_representation"])
                                temp = torch.stack(temp, dim=0)
                                temp = torch.mean(temp, dim=0)
                                
                                description_out[self.rel2id[rel]] = temp
                    # New   
                    else: 
                    # Old
                        description_out = {}
                        for rel, descriptions in seen_description.items():
                            temp = []
                            for description in descriptions:
                                des_tokens = torch.tensor([description['token_ids']]).to(args.device)
                                temp.append(encoder(des_tokens, extract_type="cls")["cls_representation"])
                            description_out[self.rel2id[rel]] = temp
                    # Old
                
                # classifier forward
                reps = classifier(encoder_out["x_encoded"])

                # prediction
                probs = F.softmax(reps, dim=1)
                _, pred = probs.max(1)
                total_hits += (pred == targets).float().sum().data.cpu().numpy().item()

                # loss components
                CE_loss = F.cross_entropy(input=reps, target=targets, reduction="mean") # cross entropy loss
                
                if args.use_ct_in_encoder == "yes":
                    if args.type_ctloss == "new":
                    # New
                        CT_loss =  new_contrastive_loss(encoder_out["x_encoded"], targets, description_out, negative_dict, args.num_descriptions) # constractive loss
                    # New
                    else:                    
                    # Old
                        CT_loss =  contrastive_loss(encoder_out["x_encoded"], targets, description_out, num_negs=args.num_negs) # constractive loss
                    # Old
                if args.use_ct_in_encoder == "yes":
                    loss = CE_loss + CT_loss*self.beta
                else:
                    loss = CE_loss
                    CT_loss = torch.tensor(0.0)
                losses.append(loss.item())
                loss.backward()
                
                # log to wandb
                wandb.log({
                    "encoder_loss": loss.item(),
                    "encoder_ce_loss": CE_loss.item(), 
                    "encoder_ct_loss": CT_loss.item()
                })

                # params update
                torch.nn.utils.clip_grad_norm_(modules_parameters, args.max_grad_norm)
                optimizer.step()

                # display
                td.set_postfix(loss=np.array(losses).mean(), acc=total_hits / sampled)


        for e_id in range(args.encoder_epochs):
            train_data(data_loader, f"train_encoder_epoch_{e_id + 1}", e_id)

    def train_prompt_pool(self, args, encoder, prompt_pool, training_data, seen_description, task_id, beta=0.1):
        encoder.eval()
        classifier = Classifier(args=args).to(args.device)
        classifier.train()
        modules = [classifier, prompt_pool]
        modules = nn.ModuleList(modules)
        modules_parameters = modules.parameters()

        optimizer = torch.optim.Adam([{"params": modules_parameters, "lr": args.prompt_pool_lr}])

        data_loader = get_data_loader(args, training_data, shuffle=True)
        new_training_data = []
        td = tqdm(data_loader, desc=f"get_prompt_key_task_{task_id+1}")
        for step, (labels, tokens, _) in enumerate(td):
            targets = labels.type(torch.LongTensor).to(args.device)
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            # encoder forward
            encoder_out = encoder(tokens)

            tokens = tokens.cpu().detach().numpy()
            x_key = encoder_out["x_encoded"].cpu().detach().numpy()
            # add to new training data
            for i in range(len(labels)):
                new_training_data.append({"relation": labels[i], "tokens": tokens[i], "key": x_key[i]})
            td.set_postfix()


        # new data loader
        data_loader = get_data_loader(args, new_training_data, shuffle=True)

        def train_data(data_loader_, name="", e_id=0):
            losses = []
            accuracies = []
            td = tqdm(data_loader_, desc=name)

            sampled = 0
            total_hits = 0

            for step, (labels, tokens, keys, _) in enumerate(td):
                optimizer.zero_grad()

                # batching
                sampled += len(labels)
                targets = labels.type(torch.LongTensor).to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                x_key = torch.stack([x.to(args.device) for x in keys], dim=0)

                # encoder forward
                encoder_out = encoder(tokens, prompt_pool, x_key)
                
                # New 
                if args.type_ctloss == "new":
                    if step % 50 ==0:
                        negative_dict = self.find_negative_labels(args, encoder, seen_description)
                    
                    all_description_label_need_cal = []
                    
                    for label in labels:
                        label = int(label)
                        if label not in all_description_label_need_cal:
                            all_description_label_need_cal.append(label)
                        for lab in negative_dict[label]:
                            if lab not in all_description_label_need_cal:
                                all_description_label_need_cal.append(lab)
                    
                    description_out = {}
                    
                    if args.strategy == 1:
                        for rel, descriptions in seen_description.items():
                            if self.rel2id[rel] in all_description_label_need_cal:
                                temp = []
                                for description in descriptions:
                                    des_tokens = torch.tensor([description['token_ids']]).to(args.device)
                                    temp.append(encoder(des_tokens, extract_type="cls")["cls_representation"])
                                temp = torch.stack(temp, dim=0)
                                temp = torch.mean(temp, dim=0)
                                
                                description_out[self.rel2id[rel]] = temp
                    elif args.strategy == 2:
                        apply_grad_list = random.sample(all_description_label_need_cal, min(args.num_grad_description_per_step, len(all_description_label_need_cal)))
                        
                        for rel, descriptions in seen_description.items():
                            if self.rel2id[rel] in all_description_label_need_cal:
                                if self.rel2id[rel] in apply_grad_list:
                                    temp = []
                                    for description in descriptions:
                                        des_tokens = torch.tensor([description['token_ids']]).to(args.device)
                                        temp.append(encoder(des_tokens, extract_type="cls")["cls_representation"])
                                    temp = torch.stack(temp, dim=0)
                                    temp = torch.mean(temp, dim=0)
                    
                                    description_out[self.rel2id[rel]] = temp   
                                else:
                                    with torch.no_grad():
                                        temp = []
                                        for description in descriptions:
                                            des_tokens = torch.tensor([description['token_ids']]).to(args.device)
                                            temp.append(encoder(des_tokens, extract_type="cls")["cls_representation"])
                                        temp = torch.stack(temp, dim=0)
                                        temp = torch.mean(temp, dim=0)
                        
                                        description_out[self.rel2id[rel]] = temp  
                    elif args.strategy == 3:
                        for rel, descriptions in seen_description.items():
                            if self.rel2id[rel] in all_description_label_need_cal:
                                temp = []
                                description = random.choice(descriptions)
                                des_tokens = torch.tensor([description['token_ids']]).to(args.device)
                                temp = encoder(des_tokens, extract_type="cls")["cls_representation"]
                                
                                description_out[self.rel2id[rel]] = temp 
                # New   
                else:
                # Old
                    description_out = {}
                    for rel, descriptions in seen_description.items():
                        temp = []
                        for description in descriptions:
                            des_tokens = torch.tensor([description['token_ids']]).to(args.device)
                            temp.append(encoder(des_tokens, extract_type="cls")["cls_representation"])
                        temp = torch.stack(temp, dim=0)
                        temp = torch.mean(temp, dim=0)
                        
                        description_out[self.rel2id[rel]] = temp
                # Old
                
                # classifier forward
                reps = classifier(encoder_out["x_encoded"])

                # prediction
                probs = F.softmax(reps, dim=1)
                _, pred = probs.max(1)
                total_hits += (pred == targets).float().sum().data.cpu().numpy().item()

                # loss components
                prompt_reduce_sim_loss = -args.pull_constraint_coeff * encoder_out["reduce_sim"]
                CE_loss = F.cross_entropy(input=reps, target=targets, reduction="mean")
                
                if args.type_ctloss == "new":
                # New
                    CT_loss =  new_contrastive_loss(encoder_out["x_encoded"], targets, description_out, negative_dict, args.num_descriptions) # constractive loss
                # New
                else:
                # Old
                    CT_loss =  contrastive_loss(encoder_out["x_encoded"], targets, description_out, num_negs=args.num_negs) # constractive loss
                # Old
                
                loss = CE_loss + prompt_reduce_sim_loss + CT_loss*self.beta
                losses.append(loss.item())
                loss.backward()
                
                # log to wandb
                wandb.log({
                    "prompt_pool_loss": loss.item(),
                    "prompt_pool_ce_loss": CE_loss.item(), 
                    "prompt_pool_ct_loss": CT_loss.item(),
                    "prompt_pool_reduce_sim_loss": prompt_reduce_sim_loss.item()
                })

                # params update
                torch.nn.utils.clip_grad_norm_(modules_parameters, args.max_grad_norm)
                optimizer.step()

                # display
                td.set_postfix(loss=np.array(losses).mean(), acc=total_hits / sampled)


        for e_id in range(args.prompt_pool_epochs):
            train_data(data_loader, f"train_prompt_pool_epoch_{e_id + 1}", e_id)

    @torch.no_grad()
    def sample_memorized_data(self, args, encoder, prompt_pool, relation_data, name, task_id):
        encoder.eval()
        data_loader = get_data_loader(args, relation_data, shuffle=False)
        td = tqdm(data_loader, desc=name)

        # output dict
        out = {}

        # x_data
        x_key = []
        x_encoded = []

        for step, (labels, tokens, _) in enumerate(td):
            try:
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                x_key.append(encoder(tokens)["x_encoded"])
                x_encoded.append(encoder(tokens, prompt_pool, x_key[-1])["x_encoded"])
            except:
                continue

        x_key = torch.cat(x_key, dim=0)
        x_encoded = torch.cat(x_encoded, dim=0)

        key_mixture = GaussianMixture(n_components=args.gmm_num_components, random_state=args.seed).fit(x_key.cpu().detach().numpy())
        encoded_mixture = GaussianMixture(n_components=args.gmm_num_components, random_state=args.seed).fit(x_encoded.cpu().detach().numpy())

        if args.gmm_num_components == 1:
            key_mixture.weights_[0] = 1.0
            encoded_mixture.weights_[0] = 1.0

        out["replay_key"] = key_mixture
        out["replay"] = encoded_mixture
        return out

    # NgoDinhLuyen EoE

    def statistic(self, args, encoder, train_data, task_id):
        for i in range(-1, task_id + 1):
            mean, cov, task_mean, task_cov = self.get_mean_and_cov(args=args, encoder=encoder, dataset=train_data, name="statistic", expert_id=i)
            self.new_statistic(args, mean, cov, task_mean, task_cov, i)
    
    def new_statistic(self, args, mean, cov, task_mean, task_cov, i):
        expert_id = i + 1
        if expert_id == 0 or expert_id == 1:
            length = self.num_tasks + 1
        else:
            length = self.num_tasks - expert_id + 2
        self.expert_distribution[expert_id]["class_mean"].append(mean.cuda())
        self.expert_distribution[expert_id]["accumulate_cov"] += cov
        avg_cov = self.expert_distribution[expert_id]["accumulate_cov"].cuda() / length
        self.expert_distribution[expert_id]["cov_inv"] = torch.linalg.pinv(avg_cov, hermitian=True)
    
    @torch.no_grad()
    def get_mean_and_cov(self, args, encoder, dataset, name, expert_id=0):
        encoder.eval()
        
        data_loader = get_data_loader(args, dataset, batch_size=1, shuffle=False)

        prelogits = []
        labels = []
        
        td = tqdm(data_loader, desc=name)
        # testing
        for step, (label, tokens, _) in enumerate(td):            
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            if expert_id == -1:
                prompted_encoder_out = self.base_bert(input_ids=tokens,
                                            attention_mask= (tokens!=0))
            elif expert_id == 0:
                prompted_encoder_out = encoder(tokens)
            else:
                # encoder forward
                encoder_out = encoder(tokens)
                
                pool_ids = [expert_id for x in range(len(label))]

                # get pools
                prompt_pools = [self.prompt_pools[x] for x in pool_ids]

                # prompted encoder forward
                prompted_encoder_out = encoder(tokens, None, encoder_out["x_encoded"], prompt_pools)

            # prediction
            prelogits.extend(prompted_encoder_out["x_encoded"].tolist())
            labels.extend(label.tolist())

        prelogits = torch.tensor(prelogits)
        labels = torch.tensor(labels)
        labels_space = torch.unique(labels)

        task_mean = prelogits.mean(dim=0)
        task_cov = torch.cov((prelogits - task_mean).T)

        mean_over_classes = []
        cov_over_classes = []
        for c in labels_space:
            embeds = prelogits[labels == c]
            if embeds.numel() > 0:
                mean = embeds.mean(dim=0)
                cov = torch.cov((embeds - mean).T)
            else:
                mean = task_mean
                cov = task_cov
            mean_over_classes.append(mean)
            cov_over_classes.append(cov)

        mean_over_classes = torch.stack(mean_over_classes)
        shared_cov = torch.stack(cov_over_classes).mean(dim=0)

        return mean_over_classes, shared_cov, task_mean, task_cov

    def get_prompt_indices(self, args, prelogits, expert_id=0):
        expert_id = expert_id + 1
        task_means_over_classes = self.expert_distribution[expert_id]["class_mean"]
        cov_inv = self.expert_distribution[expert_id]["cov_inv"]

        scores_over_tasks = []
        class_indices_over_tasks = []
        # for each task
        for idx, mean_over_classes in enumerate(task_means_over_classes):
            num_labels, _ = mean_over_classes.shape
            score_over_classes = []
            # for each label in task
            for c in range(num_labels):
                if self.query_mode == "cosine":
                    score = - F.cosine_similarity(prelogits, mean_over_classes[c])
                elif self.query_mode == "euclidean":
                    score = torch.cdist(prelogits, mean_over_classes[c].unsqueeze(0)).squeeze(1)
                elif self.query_mode == "mahalanobis":
                    score = mahalanobis(prelogits, mean_over_classes[c], cov_inv, norm=2)
                elif self.query_mode == "maha_ft":
                    score = mahalanobis(prelogits[idx], mean_over_classes[c], cov_inv, norm=2)
                else:
                    raise NotImplementedError
                score_over_classes.append(score)
            # [num_labels, n]
            score_over_classes = torch.stack(score_over_classes)
            score, class_indices = score_over_classes.min(dim=0)
            # min score of labels as task score
            scores_over_tasks.append(score)
            class_indices_over_tasks.append(class_indices + idx * num_labels)
            # class_indices_over_tasks.append(class_indices)
        # [task_num, n]
        scores_over_tasks = torch.stack(scores_over_tasks, dim=0)
        class_indices_over_tasks = torch.stack(class_indices_over_tasks, dim=0)
        _, indices = torch.min(scores_over_tasks, dim=0)
        return indices, scores_over_tasks, class_indices_over_tasks
    
    @torch.no_grad()
    def choose_indices_eoe_tii(self, args, encoder, tokens, labels, batch_size):
        encoder.eval()
        self.base_bert.eval()
        all_score_over_task = []
        all_score_over_class = []
        for e_id in range(-1, self.num_tasks + 1):
            if e_id == -1:
                hidden_states = self.base_bert(input_ids=tokens,
                                        attention_mask= (tokens!=0))
            elif e_id == 0:
                hidden_states = encoder(tokens)
            else:
                encoder_out = encoder(tokens)
                pool_ids = [e_id for x in range(len(labels))]
                prompt_pools = [self.prompt_pools[x] for x in pool_ids]
                hidden_states = encoder(tokens, None, encoder_out["x_encoded"], prompt_pools)
            
            _, scores_over_tasks, scores_over_classes = self.get_prompt_indices(args, hidden_states["x_encoded"], expert_id=e_id)
            scores_over_tasks = scores_over_tasks.transpose(-1, -2)
            scores_over_classes = scores_over_classes.transpose(-1, -2)
            if e_id != -1:
                scores_over_tasks[:, :e_id] = float('inf')  # no seen task
                # logits = self.classifier[e_id](hidden_states)[:, :self.class_per_task]
            all_score_over_task.append(scores_over_tasks)
            all_score_over_class.append(scores_over_classes)
        all_score_over_task = torch.stack(all_score_over_task, dim=1)  # (batch, expert_num, task_num)
        all_score_over_class = torch.stack(all_score_over_class, dim=1)  # (batch, expert_num, task_num)
        
        indices = []
        # expert0_score_over_task = all_score_over_task[:, 0, :]  # (batch, task_num)
        expert_values, expert_indices = torch.topk(all_score_over_task, dim=-1, k=all_score_over_task.shape[-1],
                                                    largest=False)
        expert_values = expert_values.tolist()
        expert_indices = expert_indices.tolist()
        for i in range(batch_size):
            bert_indices = expert_indices[i][0]
            task_indices = expert_indices[i][1]
            
            # if self.default_expert == "bert":
            #     default_indices = copy.deepcopy(bert_indices)
            # else:
            #     default_indices = copy.deepcopy(task_indices)
            
            default_indices = copy.deepcopy(task_indices)
            
            min_task = min(bert_indices[0], task_indices[0])
            max_task = max(bert_indices[0], task_indices[0])
            # valid_task_id = [min_task, max_task]
            cur_min_expert = min_task + 1
            if bert_indices[0] != task_indices[0] and cur_min_expert > 1:
                cur_ans = []
                for j in range(0, cur_min_expert + 1):
                    if j <= self.max_expert:  # self.max_expert==1 --> default expert
                        for k in expert_indices[i][j]:
                            if k >= min_task:
                                cur_ans.append(k)
                                break
                cur_count = Counter(cur_ans)
                most_common_element = cur_count.most_common(1)
                if most_common_element[0][1] == cur_ans.count(default_indices[0]):
                    indices.append(default_indices[0])
                else:
                    indices.append(most_common_element[0][0])
            else:
                indices.append(default_indices[0])
        
        all_score_over_class = all_score_over_class.view(all_score_over_class.shape[0], -1)
        all_score_over_class = all_score_over_class[:, 0]
        all_score_over_class = all_score_over_class.view(-1)
        new_all_score_over_class = [self.eoeid2waveid[iii] for iii in all_score_over_class.tolist()]
        new_all_score_over_class = torch.tensor(new_all_score_over_class).to(args.device)
        return indices, new_all_score_over_class
    
    @torch.no_grad()
    def choose_indices_wave(self, args, encoder, tokens, classifier):
        encoder.eval()
        encoder_out = encoder(tokens)

        # prediction
        reps = classifier(encoder_out["x_encoded"])
        probs = F.softmax(reps, dim=1)
        _, pred = probs.max(1)
        pool_ids = [self.id2taskid[int(x)] for x in pred]
        return pool_ids, pred

    # NgoDinhLuyen EoE

    @torch.no_grad()
    def evaluate_strict_model(self, args, encoder, classifier, prompted_classifier, test_data, name, task_id):
        # models evaluation mode
        encoder.eval()
        classifier.eval()
        
        # NgoDinhLuyen EoE
        batch_size = 1
        # NgoDinhLuyen EoE

        # data loader for test set
        data_loader = get_data_loader(args, test_data, batch_size=batch_size, shuffle=False)

        # tqdm
        td = tqdm(data_loader, desc=name)

        # initialization
        sampled = 0
        total_hits = np.zeros(4)

        # testing
        for step, (labels, tokens, _) in enumerate(td):
            try:
                sampled += len(labels)
                targets = labels.type(torch.LongTensor).to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)

                # NgoDinhLuyen EoE
                if args.eoe_tii == "yes":
                    pool_ids, pred  = self.choose_indices_eoe_tii(args, encoder, tokens, labels, batch_size)
                else:
                    pool_ids, pred = self.choose_indices_wave(args, encoder, tokens, classifier)
                # NgoDinhLuyen EoE

                # encoder forward
                encoder_out = encoder(tokens)

                # # prediction
                # reps = classifier(encoder_out["x_encoded"])
                # probs = F.softmax(reps, dim=1)
                # _, pred = probs.max(1)

                # accuracy_0
                total_hits[0] += (pred == targets).float().sum().data.cpu().numpy().item()

                # pool_ids
                # pool_ids = [self.id2taskid[int(x)] for x in pred]
                for i, pool_id in enumerate(pool_ids):
                    total_hits[1] += pool_id == self.id2taskid[int(labels[i])]

                # get pools
                prompt_pools = [self.prompt_pools[x] for x in pool_ids]

                # prompted encoder forward
                prompted_encoder_out = encoder(tokens, None, encoder_out["x_encoded"], prompt_pools)

                # prediction
                reps = prompted_classifier(prompted_encoder_out["x_encoded"])
                probs = F.softmax(reps, dim=1)
                _, pred = probs.max(1)

                # accuracy_2
                total_hits[2] += (pred == targets).float().sum().data.cpu().numpy().item()

                # pool_ids
                pool_ids = [self.id2taskid[int(x)] for x in labels]

                # get pools
                prompt_pools = [self.prompt_pools[x] for x in pool_ids]

                # prompted encoder forward
                prompted_encoder_out = encoder(tokens, None, encoder_out["x_encoded"], prompt_pools)

                # prediction
                reps = prompted_classifier(prompted_encoder_out["x_encoded"])
                probs = F.softmax(reps, dim=1)
                _, pred = probs.max(1)

                # accuracy_3
                total_hits[3] += (pred == targets).float().sum().data.cpu().numpy().item()

                # display
                td.set_postfix(acc=np.round(total_hits / sampled, 3))
            except:
                sampled -= len(labels)
                continue
        
        return total_hits / sampled

    def train(self, args):
        # initialize test results list
        test_cur = []
        test_total = []

        # replayed data
        self.replayed_key = [[] for e_id in range(args.replay_epochs)]
        self.replayed_data = [[] for e_id in range(args.replay_epochs)]

        # sampler
        sampler = data_sampler(args=args, seed=args.seed)
        self.rel2id = sampler.rel2id
        self.id2rel = sampler.id2rel
        
        # NgoDinhLuyen EoE
        self.eoeid2waveid = sampler.eoeid2waveid  
        print(self.eoeid2waveid)   
        # NgoDinhLuyen EoE 

        # convert
        self.id2taskid = {}

        # model
        encoder = BertRelationEncoder(config=args).to(args.device)

        # pools
        self.prompt_pools = []

        # initialize memory
        self.memorized_samples = {}

        # load data and start computation
        all_train_tasks = []
        all_tasks = []
        seen_data = {}

        for steps, (training_data, valid_data, test_data, current_relations, 
                    historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):
            
            # NgoDinhLuyen EoE
            self.num_tasks += 1
            # NgoDinhLuyen EoE
            
            print("=" * 100)
            print(f"task={steps+1}")
            print(f"current relations={current_relations}")

            self.steps = steps
            self.not_seen_rel_ids = [rel_id for rel_id in range(args.num_tasks * args.rel_per_task) if rel_id not in [self.rel2id[relation] for relation in seen_relations]]

            # initialize
            cur_training_data = []
            cur_test_data = []
            id_cur_data = []
            for i, relation in enumerate(current_relations):
                cur_training_data += training_data[relation]
                seen_data[relation] = training_data[relation]
                cur_test_data += test_data[relation]

                rel_id = self.rel2id[relation]
                self.id2taskid[rel_id] = steps
                id_cur_data.append(rel_id)

            # NgoDinhLuyen EoE
            self.expert_distribution.append({
                "class_mean": [torch.zeros(args.rel_per_task, args.encoder_output_size * 2).to(args.device) for _ in
                            range(self.num_tasks)],
                "accumulate_cov": torch.zeros(args.encoder_output_size * 2, args.encoder_output_size * 2),
                "cov_inv": torch.ones(args.encoder_output_size * 2, args.encoder_output_size * 2),
            })
            # NgoDinhLuyen EoE

            # train encoder
            if steps == 0:
                self.train_encoder(args, encoder, cur_training_data, seen_descriptions, task_id=steps, beta=args.contrastive_loss_coeff)

            # new prompt pool
            if args.use_general_pp == 1:
                self.prompt_pools.append(General_Prompt(args).to(args.device))
            else:
                self.prompt_pools.append(Prompt(args).to(args.device))
            self.train_prompt_pool(args, encoder, self.prompt_pools[-1], 
                                   cur_training_data, seen_descriptions,
                                   task_id=steps, beta=args.contrastive_loss_coeff)

            # NgoDinhLuyen EoE
            self.statistic(args, encoder, cur_training_data, steps)
            # NgoDinhLuyen EoE

            # memory
            for i, relation in enumerate(current_relations):
                self.memorized_samples[sampler.rel2id[relation]] = \
                self.sample_memorized_data(args, encoder, self.prompt_pools[steps], 
                                           training_data[relation], 
                                           f"sampling_relation_{i+1}={relation}", steps)

            # replay data for classifier
            for relation in current_relations:
                print(f"replaying data {relation}")
                rel_id = self.rel2id[relation]
                replay_data = self.memorized_samples[rel_id]["replay"].sample(args.replay_epochs * args.replay_s_e_e)[0].astype("float32")
                for e_id in range(args.replay_epochs):
                    for x_encoded in replay_data[e_id * args.replay_s_e_e : (e_id + 1) * args.replay_s_e_e]:
                        self.replayed_data[e_id].append({"relation": rel_id, "tokens": x_encoded})

            for relation in current_relations:
                print(f"replaying key {relation}")
                rel_id = self.rel2id[relation]
                replay_key = self.memorized_samples[rel_id]["replay_key"].sample(args.replay_epochs * args.replay_s_e_e)[0].astype("float32")
                for e_id in range(args.replay_epochs):
                    for x_encoded in replay_key[e_id * args.replay_s_e_e : (e_id + 1) * args.replay_s_e_e]:
                        self.replayed_key[e_id].append({"relation": rel_id, "tokens": x_encoded})

            # all
            all_train_tasks.append(cur_training_data)
            all_tasks.append(cur_test_data)

            # evaluates
            need_evaluates = list(range(1, 11))
            if steps + 1 in need_evaluates:
                # classifier
                classifier = Classifier(args=args).to(args.device)
                swag_classifier = SWAG(Classifier, no_cov_mat=not (args.cov_mat), max_num_models=args.max_num_models, args=args)

                # classifier
                prompted_classifier = Classifier(args=args).to(args.device)
                swag_prompted_classifier = SWAG(Classifier, no_cov_mat=not (args.cov_mat), max_num_models=args.max_num_models, args=args)

                # train
                self.train_classifier(args, classifier, swag_classifier, self.replayed_key, "train_classifier_epoch_")
                self.train_classifier(args, prompted_classifier, swag_prompted_classifier, self.replayed_data, "train_prompted_classifier_epoch_")

                # prediction
                print("===NON-SWAG===")
                results = []
                for i, i_th_test_data in enumerate(all_tasks):
                    results.append([
                        len(i_th_test_data), 
                        self.evaluate_strict_model(args, encoder, classifier, prompted_classifier, 
                                                    i_th_test_data, f"test_task_{i+1}", steps)
                    ])
                cur_acc = results[-1][1]
                total_acc = sum([result[0] * result[1] for result in results]) / sum([result[0] for result in results])
                print(f"current test accuracy: {cur_acc}")
                print(f"history test accuracy: {total_acc}")
                test_cur.append(cur_acc)
                test_total.append(total_acc)

                print("===SWAG===")
                results = []
                for i, i_th_test_data in enumerate(all_tasks):
                    results.append([
                        len(i_th_test_data), 
                        self.evaluate_strict_model(args, encoder, swag_classifier, swag_prompted_classifier, 
                                                   i_th_test_data, f"test_task_{i+1}", steps)
                    ])
                cur_acc = results[-1][1]
                total_acc = sum([result[0] * result[1] for result in results]) / sum([result[0] for result in results])
                print(f"current test accuracy: {cur_acc}")
                print(f"history test accuracy: {total_acc}")
                test_cur.append(cur_acc)
                test_total.append(total_acc)

                acc_sum =[]
                print("===UNTIL-NOW===")
                print("accuracies:")
                for x in test_cur:
                    print(x)
                print("arverages:")
                for x in test_total:
                    print(x)
                    acc_sum.append(x)
                    
                results.append({
                    "task": steps,
                    "results": list(acc_sum),
                })
                
                # Tạo thư mục lưu kết quả dựa trên seed và các tham số quan trọng
                result_dir = f"./results/{args.dataname}_seed{args.seed}_encLR{args.encoder_lr}_clsLR{args.classifier_lr}_promptLen{args.prompt_length}_pull{args.pull_constraint_coeff}_contrast{args.contrastive_loss_coeff}"
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                # Lưu kết quả với tên file chứa thông tin về bước hiện tại
                result_file = f"{result_dir}/task_{steps}.pickle"
                with open(result_file, "wb") as file:
                    pickle.dump(results, file)


        del self.memorized_samples, 
        self.prompt_pools, all_train_tasks, 
        all_tasks, seen_data, results, encoder, 
        self.id2taskid, sampler, self.replayed_data, 
        self.replayed_key, test_cur, test_total