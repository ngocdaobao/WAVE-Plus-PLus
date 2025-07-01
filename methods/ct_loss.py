import torch
import torch.nn.functional as F

def sim(x, y):
    """
    Tính độ tương đồng giữa hai vectơ x, y
    
    - x: Tensor (N, D), batch của N vectơ đầu vào
    - y: Tensor (M, D), batch của M vectơ so sánh
    
    Trả về:
    - sim: Tensor (N, M), ma trận độ tương đồng giữa x và y
    """
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    
    return torch.mm(x, y.t())

def contrastive_loss(reps, targets, descriptions, num_negs = 4, temperature=5):
    """
    Tính loss kiểu -log(sim(x, des(x)) / sim(x, des))
    
    - reps: Tensor (N, D), biểu diễn đặc trưng của các mẫu
    - targets: Tensor (N,), nhãn tương ứng của reps
    - descriptions: Dict[int, Tensor], ánh xạ nhãn đến mô tả (M, D)
    - temperature: Hệ số nhiệt độ để điều chỉnh độ sắc nét của phân phối
    
    Trả về:
    - loss: Giá trị tổn thất trung bình
    """
    device = reps.device
    
    # Tạo batch descriptions tương ứng với từng mẫu trong reps
    desc_list = torch.stack([descriptions[int(label)][0][0] for label in targets]).to(device)  # (N, D)
    
    # Tạo batch tất cả descriptions
    all_descriptions = torch.stack([des[0][0] for des in descriptions.values()]).to(device)  # (M, D)
    
    # Tính cosine similarity giữa reps và descriptions
    similarities = sim(reps, all_descriptions) / temperature  # (N, M)
    
    # Lấy similarity giữa reps và mô tả tương ứng
    pos_sim = sim(reps, desc_list).diag()  # (N,)
    
    # Lấy top-k mô tả gần nhất với reps
    num_negs = min(num_negs, similarities.size(1))
    _, negs = similarities.topk(num_negs, dim=1) # (N, num_negs)

    
    # Lấy similarity giữa reps và mô tả ngẫu nhiên
    neg_sims = similarities[torch.arange(len(targets)).unsqueeze(1), negs]  # (N, num_negs)
    
    # Tính loss theo công thức -log(sim(x, des(x)) / (sim(x, des(x) + sim(x, neg_des)))
    loss = -torch.log(torch.sigmoid(pos_sim.unsqueeze(1) - neg_sims).mean(dim=1)).mean()
    
    return loss.mean()

def new_contrastive_loss(reps, targets, descriptions, negative_dict, num_description_per_label, temperature=5):
    """
    Tính loss kiểu -log(sim(x, des(x)) / sim(x, des))
    
    - reps: Tensor (N, D), biểu diễn đặc trưng của các mẫu
    - targets: Tensor (N,), nhãn tương ứng của reps
    - descriptions: Dict[int, Tensor], ánh xạ nhãn đến mô tả (M, D)
    - temperature: Hệ số nhiệt độ để điều chỉnh độ sắc nét của phân phối
    
    Trả về:
    - loss: Giá trị tổn thất trung bình
    """
    device = reps.device
        
    # Tạo batch descriptions tương ứng với từng mẫu trong reps
    desc_list = torch.stack([descriptions[int(label)][0] for label in targets]).to(device)  # (N, D)
    
    # Tạo batch tất cả descriptions
    idx2idmatrix = {}
    all_descriptions = []
    for idx, (id_rel, embed) in enumerate(descriptions.items()):
        all_descriptions.append(embed[0])
        idx2idmatrix[id_rel] = idx
    all_descriptions = torch.stack(all_descriptions, dim=0).to(device)
    
    # Tính cosine similarity giữa reps và descriptions
    similarities = sim(reps, all_descriptions) / temperature  # (N, M)
    
    # Lấy similarity giữa reps và mô tả tương ứng
    pos_sim = sim(reps, desc_list).diag()  # (N,)
    
    expanded_negs = []
    for label in targets:
        neg_indices = []
        for neg_label in negative_dict[int(label)]:  # Duyệt qua negative labels
            neg_indices.append(idx2idmatrix[neg_label])  # Lấy tất cả index của negative label
        expanded_negs.append(neg_indices)

    # Chuyển thành tensor
    negs = torch.tensor(expanded_negs, device=device)
    
    # Lấy similarity giữa reps và mô tả ngẫu nhiên
    neg_sims = similarities[torch.arange(len(targets)).unsqueeze(1), negs] # (N, num_negs)
    
    # Tính loss theo công thức -log(sim(x, des(x)) / (sim(x, des(x) + sim(x, neg_des)))
    loss = -torch.log(torch.sigmoid(pos_sim.unsqueeze(1) - neg_sims).mean(dim=1)).mean()
    
    
    return loss.mean()
