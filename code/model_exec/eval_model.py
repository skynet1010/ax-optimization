import torch
from utils.consts import nr_of_classes, loss_dict
from typing import Dict


def evaluate(
    model: torch.nn.Module,
    eval_data_loader: torch.utils.data.DataLoader,
    criterion:torch.nn.modules.loss._Loss,
    device: torch.device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype=torch.float
) -> Dict:
    model.to(device=device,dtype=dtype)

    model.eval()

    correct = 0
    total = 0
    running_loss=0

    softmax = torch.nn.Softmax(dim=1)

    tp = 0
    fn = 0
    fp = 0
    tn = 0
    tp_c = 0
    fp_c = 0
    tn_c = 0
    fn_c = 0

    with torch.no_grad():
        for data in eval_data_loader:
            tmp_batch_size = len(data["labels"])
            lbl_onehot = torch.FloatTensor(tmp_batch_size,2).to(device=device,dtype=dtype)
            # =============datapreprocessing=================
            img = torch.FloatTensor(data["imagery"]).to(device=device,dtype=dtype)
            # ===================forward=====================
            
            output = model(img)
            
            out_softmax = softmax(output)

            lbl_onehot.zero_()
            loss = criterion(output, lbl_onehot)

            running_loss+=(loss.item()*tmp_batch_size)
            confidence, predicted = torch.max(out_softmax.data, 1)
            total += tmp_batch_size

            labels = data["labels"].view(tmp_batch_size)
            pred_cpu = predicted.cpu()
            correct += (pred_cpu == labels).sum().item()

            label_ones_idx = labels.nonzero()
            label_zeroes_idx = (labels==0).nonzero()
            tp_idx = pred_cpu[label_ones_idx]==labels[label_ones_idx]
            tp += (tp_idx).sum().item()
            fp_idx = pred_cpu[label_ones_idx]!=labels[label_ones_idx]
            fp += (fp_idx).sum().item()
            tn_idx = pred_cpu[label_zeroes_idx]==labels[label_zeroes_idx]
            tn += (tn_idx).sum().item()
            fn_idx = pred_cpu[label_zeroes_idx]!=labels[label_zeroes_idx] 
            fn += (fn_idx).sum().item()
            tp_c += confidence[tp_idx].sum().item()
            fp_c += confidence[fp_idx].sum().item()
            tn_c += confidence[tn_idx].sum().item()
            fn_c += confidence[fn_idx].sum().item()

    metrics = {"acc":correct/total, "loss":running_loss/total,"TP":tp,"FN":fn,"FP":fp,"TN":tn,"TPC":tp_c/total,"FPC":fp_c/total,"TNC":tn_c/total,"FNC":fn_c/total}
    return metrics