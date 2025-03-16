import os
import argparse


class Param:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()
        for unknown_arg in unknown:
            print(f"Unknown argument: {unknown_arg}")
        self.args = all_args

    def all_param(self, parser):
        # eoe tii
        parser.add_argument("--eoe_tii", default="yes", type=str)
        parser.add_argument("--num_descriptions", default=1, type=int)
        parser.add_argument("--type_ctloss", default="new", type=str)
        parser.add_argument("--use_ct_in_encoder", default="yes", type=str)
        parser.add_argument("--beta", default=0.1, type=float)
        parser.add_argument("--num_grad_description_per_step", default=3, type=int)
        parser.add_argument("--strategy", default=3, type=int)
        parser.add_argument("--use_general_pp", default=0, type=int)
        parser.add_argument("--weight_general_prompt", default=0.5, type=float)
        
        # common parameters
        parser.add_argument("--gpu", default=0, type=int)
        parser.add_argument("--dataname", default="FewRel", type=str, help="Use TACRED or FewRel datasets.")
        parser.add_argument("--task_name", default="FewRel", type=str)
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--run_name", default=None, type=str)

        # training parameters
        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--num_tasks", default=10)
        parser.add_argument("--rel_per_task", default=8)
        parser.add_argument("--pattern", default="entity_marker")
        parser.add_argument("--max_length", default=256, type=int)
        parser.add_argument("--encoder_output_size", default=768, type=int)
        parser.add_argument("--vocab_size", default=30522, type=int)
        parser.add_argument("--marker_size", default=4, type=int)
        parser.add_argument("--num_workers", default=0, type=int)

        # learning rate
        parser.add_argument("--classifier_lr", default=1e-2, type=float)
        parser.add_argument("--encoder_lr", default=1e-3, type=float)
        parser.add_argument("--prompt_pool_lr", default=1e-3, type=float)
        # momentum
        parser.add_argument("--sgd_momentum", default=0.1, type=float)

        # gmm
        parser.add_argument("--gmm_num_components", default=1, type=int)
        # loss balancing
        parser.add_argument("--pull_constraint_coeff", default=0.1, type=float)
        parser.add_argument("--contrastive_loss_coeff", default=0.1, type=float)
        parser.add_argument("--num_negs", default=4, type=int)

        # epochs
        parser.add_argument("--classifier_epochs", default=100, type=int)
        parser.add_argument("--encoder_epochs", default=10, type=int)
        parser.add_argument("--prompt_pool_epochs", default=10, type=int)

        # replay size
        parser.add_argument("--replay_s_e_e", default=256, type=int)
        parser.add_argument("--replay_epochs", default=100, type=int)

        # seed
        parser.add_argument("--seed", default=2021, type=int)
        # max gradient norm
        parser.add_argument("--max_grad_norm", default=10, type=float)

        # dataset path
        parser.add_argument("--data_path", default="datasets/", type=str)
        # bert-base-uncased weights path
        parser.add_argument("--bert_path", default="datasets/bert-base-uncased", type=str)

        # swag params
        parser.add_argument("--cov_mat", action="store_false", default=True)
        parser.add_argument("--max_num_models", type=int, default=10)
        parser.add_argument("--sample_freq", type=int, default=20)

        # prompt params
        parser.add_argument("--prompt_length", type=int, default=1)
        parser.add_argument("--prompt_embed_dim", type=int, default=768)
        parser.add_argument("--prompt_pool_size", type=int, default=80)
        parser.add_argument("--prompt_top_k", type=int, default=8)
        parser.add_argument("--prompt_init", default="uniform", type=str)
        parser.add_argument("--prompt_key_init", default="uniform", type=str)
        parser.add_argument("--prompt-type", default="coda-prompt", type=str)

        return parser
