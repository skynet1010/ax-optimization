from argparse import Namespace
from psycopg2.extensions import connection

import torch
from torch import nn
import numpy as np

from ax.service.managed_loop import optimize
from typing import Dict
import time
import os
import sys
import json
from ax import save

from utils.dataloader_provider import get_dataloaders
from utils.postgres_functions import insert_row,  make_sure_table_exist
from utils.consts import optimizer_dict, loss_dict, data_compositions
from utils.model_manipulator import manipulateModel
from model_exec.eval_model import evaluate
from model_exec.train_model import train
from utils.metric_monitor import calc_metrics

conn=None
cur = None
args = None
ss = None
data_composition_key = None
model_key = None
task = None
res_path = None

best_loss = sys.float_info.max
best_acc = 0.0
best_exec_time = sys.float_info.max

def get_results_dir(args):
    return os.path.join(args.results_dir,args.run_dir)

def get_valid_path(args,data_composition_key,ss):
    res_path = os.path.join(get_results_dir(args),f"{data_composition_key}",f"{ss}")
    if not os.path.isdir(res_path):
        try:
            os.makedirs(res_path)
        except Exception as e:
            print(e)
            exit(1)
    return res_path



def objective(parameters:Dict):
    parameters_str = str(parameters).replace("'","*")
    print(parameters_str)
    global res_path
    res_path = get_valid_path(args,data_composition_key,ss)

    best_checkpoint_path = os.path.join(res_path,f"best_{model_key}_final.pth")
    best_param_config_path = os.path.join(res_path,f"best_{model_key}_param_config.json")

    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(args,ss,data_composition_key, model_key)
    model = manipulateModel(model_key,parameters.get("feature_extraction",True),data_compositions[data_composition_key])
    
    criterion = loss_dict[parameters.get("criterion","MSELoss")]()
    optimizer = optimizer_dict[parameters.get("optimizer","Adam")](model.parameters(), lr=parameters.get("lr",1e-3),weight_decay=parameters.get("weight_decay",1e-5))
    
    global best_loss
    global best_acc
    global best_exec_time

    update = False
    no_improve_it = 0
    for epoch in range(1,args.epochs+1):
        try:
            start = time.time()
            model,train_metrics = train(model,train_data_loader,optimizer,criterion) 
            train_metrics = calc_metrics(train_metrics)
            train_exec_time = time.time()-start

            start = time.time()
            valid_metrics =  evaluate(model,valid_data_loader,criterion)
            valid_metrics = calc_metrics(valid_metrics)
            valid_exec_time = time.time()-start

            train_metrics["exec_time"] = train_exec_time
            valid_metrics["exec_time"] = valid_exec_time
            if valid_metrics["loss"] < best_loss:
                best_acc = valid_metrics["acc"]
                best_loss = valid_metrics["loss"]
                update=True
            elif valid_metrics["loss"] == best_loss and best_acc > valid_metrics["acc"]:
                best_acc = valid_metrics["acc"]
                update=True
            elif valid_metrics["acc"] == best_acc and best_loss == valid_metrics["loss"] and valid_exec_time<best_exec_time:
                update=True
            if update:
                no_improve_it = 0
                best_exec_time = valid_exec_time
                torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}, best_checkpoint_path)
                with open(best_param_config_path,"w")as f:
                    json.dump(parameters,f)
                update=False
            else:
                no_improve_it+=1
            cur.execute(insert_row(args.train_results_ax_table_name,args, task,parameters_str,epoch,timestamp=time.time(),m=train_metrics))
            conn.commit()
            cur.execute(insert_row(args.validation_results_ax_table_name,args, task,parameters_str,epoch,timestamp=time.time(),m=valid_metrics))
            conn.commit()

            print('epoch [{}/{}], loss:{:.4f}, acc: {:.4f}%, time: {} s'.format(epoch, args.epochs, valid_metrics["loss"],valid_metrics["acc"]*100, train_exec_time+valid_exec_time))        
            if no_improve_it == args.earlystopping_it:
                break
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, halfing batch size')
                args.batch_size=max(1,args.batch_size//2)
                print(f"Batch size is now: {args.batch_size}")
                return objective(parameters)
            else:
                print("Another RuntimeError occured during train an validation", e)
                exit(1) 
        except Exception as e:
            print(e)
            exit(1)
    model = manipulateModel(model_key,args.is_feature_extraction,data_compositions[data_composition_key])
    if not os.path.isfile(best_checkpoint_path):
        print("Best checkpoint file does not exist!!!")
        return True
    
    best_checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    optimizer.load_state_dict(best_checkpoint["optimizer_state_dict"])
    start = time.time()
    test_metrics = evaluate(model,test_data_loader,criterion)

    test_metrics = calc_metrics(test_metrics)
    test_metrics["exec_time"] = time.time()-start
    
    cur.execute(insert_row(args.test_results_ax_table_name,args, task,parameters_str,epoch,timestamp=time.time(),m=test_metrics))
    conn.commit()

    return best_acc

def hyperparameter_optimization(a:Namespace,c:connection,t:str):
    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global cur
    cur = c.cursor()
    global conn
    conn = c
    global args
    args = a
    global task
    task = t

    global ss
    global data_composition_key
    global model_key
    _,ss,data_composition_key,model_key,ntrails,epochs=task.split(":")
    args.epochs = 2#int(epochs)

    make_sure_table_exist(args, conn, cur, args.train_results_ax_table_name)
    make_sure_table_exist(args, conn, cur, args.validation_results_ax_table_name)
    make_sure_table_exist(args, conn, cur, args.test_results_ax_table_name)

    objective({}) #initial run config

    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-7, 0.5], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-8, .5],"log_scale": True},
            {"name":"optimizer","type":"choice", "values":["Adadelta","Adagrad","Adam","AdamW","Adamax","ASGD","RMSprop","SGD"]},
            {"name":"criterion","type":"choice", "values":["BCELoss","MSELoss"]},
            {"name":"feature_extraction","type":"choice", "values":[True,False]}
        ],
        evaluation_function=objective,
        objective_name='accuracy',
        minimize=False,
        arms_per_trial=1,
        total_trials=2#int(ntrails)#<---------------------------anpassen je nach task =)
    )
    
    save(experiment,os.path.join(res_path,"experiment.json"))
        
    return True