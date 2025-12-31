import yaml
import torch
from config import Param
from methods.utils import setup_seed
from methods.manager import Manager
import os

import logging
import sys
import re
from datetime import datetime

# Lớp ghi log để ghi cả print() vào file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.tqdm_pattern = re.compile(r'\r.*')  # Loại bỏ các dòng tqdm

    def write(self, message):
        if not self.tqdm_pattern.match(message):  # Bỏ qua tqdm
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run(args):
    print(f"hyper-parameter configurations:")
    print(yaml.dump(args.__dict__, sort_keys=True, indent=4))

    setup_seed(args.seed)
    
    # for seed in [2421, 2021]:
        # args.seed = seed
    manager = Manager(args)
    manager.train(args)


if __name__ == "__main__":
    # Load configuration
    param = Param()
    args = param.args
    
    # ensure run name exists
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Thêm timestamp
        args.run_name = f"{args.dataname}_{args.seed}_{args.num_descriptions}_{args.prompt_pool_size}_{args.prompt_length}_{args.prompt_top_k}_{timestamp}"

        
    # Cấu hình logging
    log_dir = "log_result"
    os.makedirs(log_dir, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại
    log_filename = os.path.join(log_dir, f"{args.run_name}_log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="w"),  # Ghi vào file
            logging.StreamHandler(sys.stdout)  # Hiển thị trên terminal
        ]
    )
    
    # Ghi cả print() vào file log
    sys.stdout = Logger(log_filename)
    sys.stderr = sys.stdout  # Để ghi cả lỗi vào file

    logging.info("Wandb support removed. Running without wandb logging.")

    # Device
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.device)

    # Num GPU
    args.n_gpu = torch.cuda.device_count()

    # Task name
    args.task_name = args.dataname

    # rel_per_task
    if args.dataname == "FewRel":
        args.rel_per_task = 8
    elif args.dataname == "TACRED":
        args.rel_per_task = 4
    else:
        raise ValueError("Invalid dataname")

    # Run
    run(args)
    
    logging.info("Run finished.")
