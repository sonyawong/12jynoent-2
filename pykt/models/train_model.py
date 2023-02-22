import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from ..utils.utils import debug_print
from pykt.config import que_type_models
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def cal_loss(model, ys, r, rshft, sm, preloss=[]):
    model_name = model.model_name

    y = torch.masked_select(ys[0], sm)
    t = torch.masked_select(rshft, sm)
    # print(f"loss1: {y.shape}")
    loss1 = binary_cross_entropy(y.double(), t.double())

    if model.emb_type.find("predcurc") != -1:
        if model.emb_type.find("his") != -1:
            loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
        else:
            loss = model.l1*loss1+model.l2*ys[1]
    elif model.emb_type.find("predhis") != -1:
        loss = model.l1*loss1+model.l2*ys[1]
    else:
        loss = loss1
    
    return loss


def model_forward(model, data, attn_grads=None):
    model_name = model.model_name
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
    dcur = data
    q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]
    qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
    m, sm = dcur["masks"], dcur["smasks"]

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)

    y, y2, y3 = model(dcur, train=True, attn_grads=attn_grads)
    ys = [y[:,1:], y2, y3]

    loss = cal_loss(model, ys, r, rshft, sm, preloss)

    return loss
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, dataset_name=None, fold=None):
    max_auc, best_epoch = 0, -1
    train_step = 0
    if model.model_name=='lpkt':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    for i in range(1, num_epochs + 1):
        loss_mean = []
        for j,data in enumerate(train_loader):
            train_step+=1
            if model.model_name in que_type_models:
                model.model.train()
            else:
                model.train()
            if model.model_name.find("bakt") != -1:
                if j == 0 or model.emb_type.find("grad") == -1 and model.emb_type != "qid":attn_grads=None
                # if model.model_name.find("qikt") == -1:
                #     if j != 0:pre_attn_weights = model.attn_weights
                loss = model_forward(model, data, attn_grads)
            else:
                loss = model_forward(model, data, i)
            opt.zero_grad()
            loss.backward()#compute gradients            
            opt.step()#update model’s parameters
                
            loss_mean.append(loss.detach().cpu().numpy())
            if model.model_name == "gkt" and train_step%10==0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text = text,fuc_name="train_model")
        if model.model_name=='lpkt':
            scheduler.step()#update each epoch
        loss_mean = np.mean(loss_mean)
        
        auc, acc = evaluate(model, valid_loader, model.model_name)
        ### atkt 有diff， 以下代码导致的
        ### auc, acc = round(auc, 4), round(acc, 4)

        if auc > max_auc+1e-3:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+"_model.ckpt"))
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            validauc, validacc = auc, acc
        print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")


        if i - best_epoch >= 10:
            break
    if model.emb_type == "qid":
        attn_grads = attn_grads.detach().cpu().numpy()
        # print(f"writing:{attn_grads.shape}")
        np.savez(f"./save_attn/{dataset_name}_save_grad_fold_{fold}.npz",attn_grads)
        attn_weights = attn_weights.detach().cpu().numpy()
        # print(f"attn_weights:{model.attn_weights.shape}")        
        np.savez(f"./save_attn/{dataset_name}_save_attnweight_fold_{fold}.npz",attn_weights)
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch
