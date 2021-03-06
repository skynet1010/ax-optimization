import torch
from typing import Dict
from utils.consts import nr_of_classes, loss_dict,optimizer_dict

def train(
    model: torch.nn.Module,
    train_data_loader: torch.utils.data.DataLoader, 
    optimizer:torch.optim.Optimizer,
    criterion:torch.nn.modules.loss._Loss,
    device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype=torch.float,
) -> torch.nn.Module:
    model.to(device=device,dtype=dtype)
    model.train()
   
    running_loss = 0
    correct= 0
    total=0

    softmax = torch.nn.Softmax(dim=1)

    tp = 0
    fn = 0
    fp = 0
    tn = 0
    tp_c = 0
    fp_c = 0
    tn_c = 0
    fn_c = 0

    for step,data in enumerate(train_data_loader):
        tmp_batch_size = len(data["labels"])
        lbl_onehot = torch.FloatTensor(tmp_batch_size,2).to(device=device,dtype=dtype)
        # =============datapreprocessing=================
        img = torch.FloatTensor(data["imagery"].float()).to(device=device,dtype=dtype)
        lbl_onehot.zero_()
        lbl_onehot = lbl_onehot.scatter(1,data["labels"].to(device=device,dtype=torch.long),1).to(device=device,dtype=dtype)
        # ===================forward=====================
        output = model(img)
        
        out_softmax = softmax(output)

        loss = criterion(out_softmax, lbl_onehot)

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        running_loss+=(loss.item()*tmp_batch_size)

        out_softmax = softmax(output)

        confidence, predicted = torch.max(out_softmax, 1)
        total += tmp_batch_size
        labels = data["labels"].view(tmp_batch_size)
        pred_cpu = predicted.cpu()
        correct += (pred_cpu == labels).sum().item()

        label_ones_idx = torch.squeeze(labels.nonzero())
        label_zeroes_idx = torch.squeeze((labels==0).nonzero())

        tp_idx = (pred_cpu[label_ones_idx]==labels[label_ones_idx]).nonzero()
        tp += tp_idx.size()[0]

        fp_idx = (pred_cpu[label_ones_idx]!=labels[label_ones_idx]).nonzero()
        fp += fp_idx.size()[0]

        tn_idx = (pred_cpu[label_zeroes_idx]==labels[label_zeroes_idx]).nonzero()
        tn += tn_idx.size()[0]

        fn_idx = (pred_cpu[label_zeroes_idx]!=labels[label_zeroes_idx]).nonzero()
        fn += fn_idx.size()[0]

        tp_c += confidence[tp_idx].sum().item()
        fp_c += confidence[fp_idx].sum().item()
        tn_c += confidence[tn_idx].sum().item()
        fn_c += confidence[fn_idx].sum().item()

    metrics = {"acc":correct/total, "loss":running_loss/total,"TP":tp,"FN":fn,"FP":fp,"TN":tn,"AVG_TPC":tp_c/max(1,tp),"AVG_FPC":fp_c/max(1,fp),"AVG_TNC":tn_c/max(1,tn),"AVG_FNC":fn_c/max(1,fn)}

    return model,metrics