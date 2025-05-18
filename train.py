# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from loss.focal_loss import MultiFocalLoss
from utils_GateKeeper import get_time_dif
import wandb

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


        
def train(config, model, train_iter, dev_iter, test_iter, data):
        
  
    print('num class is ', config.num_classes)

    print(config.train_path.split("\\")[-2])
    wandb.init(project=config.model_name+"-"+config.train_path.split("\\")[-3])
    wandb.config = {
    "learning_rate": config.learning_rate,
    "epochs": config.num_epochs,
    "batch_size": config.batch_size
    }

    Loss = MultiFocalLoss(num_class= config.num_classes, gamma=2.0, reduction='mean')      #config.num_classes
    
    start_time = time.perf_counter()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lambda1 = lambda epoch:np.sin(epoch)/epoch
 #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda1)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    
    for epoch in range(config.num_epochs): 
        print('Epoch [{}/{}]'.format(epoch + 1,config.num_epochs))
        #print(np.shape(train_iter))

        for i,(traffic, pos, labels) in enumerate(train_iter): 
            
            #print(traffic.size())
            # print(pos.size())
            #print(labels.size())
            preds,_ = model(traffic, pos)
            #loss = F.cross_entropy(preds, labels)
            loss = Loss(preds,labels)
           
            optimizer.zero_grad()               
            loss.backward()       
            optimizer.step()
            #scheduler.step()       

          
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(preds.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                    
                else:
                    improve = ''

                wandb.log({"train_loss":  loss.item()})
                wandb.log({"train_acc":  train_acc})
                
                model.train()
                wandb.watch(model)
                
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, {5}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc,improve))
             
            total_batch += 1
            if total_batch - last_improve > 200000:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        
        if flag:
            break
    end_time = time.perf_counter()
    time_dif = end_time - start_time
    time_dif = time_dif/60
    average_time = time_dif/config.num_epochs
    print(f"Training Time : {time_dif:.2f} Minutes")  # Show 6 decimal places
    print(f"Average Training time (epoch): {average_time:.2f} Minutes")
    wandb.log({"training_time":  float(time_dif)})
    wandb.log({"average_training_time":  float(average_time)})
    
    test(config, model, test_iter, data)



def test(config, model, test_iter, data):
    # test

    wandb.init(project=config.model_name+"-"+config.train_path.split("\\")[-3]+"-test")
    wandb.config = {
    "learning_rate": config.learning_rate,
    "epochs": config.num_epochs,
    "batch_size": config.batch_size
    }

    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    wandb.watch(model, log_graph=True)
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True, data=data)
    wandb.log({"test_loss":  test_loss})
    wandb.log({"test_acc":  test_acc})
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    

def evaluate(config, model, data_iter, test=False, data=None):
    model.eval()
    start_time = time.time()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    Loss = MultiFocalLoss(num_class=config.num_classes, gamma=2.0, reduction='mean')
    
    with torch.no_grad():
        for traffic, pos, labels in data_iter:
            
            outputs,_ = model(traffic,pos)
            loss = F.cross_entropy(outputs, labels)
            #loss = Loss(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()

            predict_ = torch.softmax(outputs,dim=1)
            predict_ = predict_.cpu().numpy()

            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    

    if test == True:
        time_dif, average_time = get_time_dif(start_time, test=1, data=data)
        print(f"Testing Time usage: {time_dif:.10f} seconds")  
        print(f"Average Testing time: {average_time:.10f} seconds")
        wandb.log({"test_time":  float(time_dif)})
        wandb.log({"average_time":  float(average_time)})

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
