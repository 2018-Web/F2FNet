import torch
from torch import optim as optim
import torch.optim as optim
import torch.nn.functional as F
import time
import warnings
import neptune.new as neptune
import numpy as np
from Utils.avg_meter import AverageMeter
from Utils.Metrics import accuracy,kappa,confusion_matrix,g_mean
import matplotlib.pyplot as plt
import os
from torch.utils.checkpoint import checkpoint
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
warnings.filterwarnings("ignore")
# Training the supervised model
def supervised_training(Net, train_data_loader,val_data_loader, criterion, optimizer, args, device):
    warnings.filterwarnings("ignore")

    if args.is_neptune:
        run = neptune.init(project= "mithunjha/earEEG-v2-cross", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZjA0YTVhOC02ZGVlLTQ0NTktOWY3NS03YzFhZWUxY2M4MTcifQ==")
        parameters = {"Experiment" : "Supervised training", 
        'Loss': "Categorical Crossentropy Loss",
        'Model Type' : args.model,
        'd_model' : args.d_model,
        'depth' : args.depth,
        'dim_feedforward' : args.dim_feedforward, 
        'window_size ':args.window_size ,
        'Batch Size': args.batch_size,
        'Optimizer' : "Adam",
        'Learning Rate': args.lr,
        'eps' : args.eps,
        'Beta 1': args.beta_1,
        'Beta 2': args.beta_2,
        'n_epochs': args.n_epochs,
        'val_set' : args.val_data_list[0]+1}
        run['model/parameters'] = parameters
        run['model/model_architecture'] = Net

    best_val_acc = 0.0
    best_val_kappa = 0.0
    for epoch_idx in range(args.n_epochs):  # loop over the dataset multiple times
        Net.train()
        print(f'===========================================================Training Epoch : [{epoch_idx+1}/{args.n_epochs}] ===========================================================================================================>')
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        losses = AverageMeter()
        val_losses = AverageMeter()
        
        train_accuracy = AverageMeter()
        val_accuracy = AverageMeter()

        train_sensitivity = AverageMeter()
        val_sensitivity = AverageMeter()
        
        train_specificity = AverageMeter()
        val_specificity = AverageMeter()

        train_gmean = AverageMeter()
        val_gmean = AverageMeter()

        train_kappa = AverageMeter()
        val_kappa = AverageMeter()

        train_f1_score = AverageMeter()
        val_f1_score = AverageMeter()

        train_precision = AverageMeter()
        val_precision = AverageMeter()

        class1_sens = AverageMeter()
        class2_sens = AverageMeter()
        class3_sens = AverageMeter()
        class4_sens = AverageMeter()
        class5_sens = AverageMeter()

        class1_spec = AverageMeter()
        class2_spec = AverageMeter()
        class3_spec = AverageMeter()
        class4_spec = AverageMeter()
        class5_spec = AverageMeter()

        class1_f1 = AverageMeter()
        class2_f1 = AverageMeter()
        class3_f1 = AverageMeter()
        class4_f1 = AverageMeter()
        class5_f1 = AverageMeter()

        end = time.time()

        for batch_idx, data_input in enumerate(train_data_loader):
            # get the inputs; data is a list of [inputs, labels]
            data_time.update(time.time() - end)
            # 数据加载
            # 获取 EEG adj labels
            psg, labels = data_input
            eeg = psg[:,:,0,:]
            eog = psg[:,:,1,:]
            emg = psg[:,:,2,:]
            cur_batch_size = len(eeg)
            

            optimizer.zero_grad()

            if args.model == 'CMT':
                outputs,_,_ = Net(eeg.float().to(device), eog.float().to(device), emg.float().to(device))
            elif args.model == 'USleep':
                outputs,_ = Net(psg.float().to(device))

            if args.model == 'GNN':
                print()
                # outputs = Net(eeg.float().to(device),adj.float().to(device))
            elif args.model == 'EEGNet':
                outputs = Net(eeg.float().to(device))

            loss = criterion(outputs.cpu(), labels)#.to(device))
            loss.backward()
            optimizer.step()
        
            
            losses.update(loss.data.item())
            train_accuracy.update(accuracy(outputs.cpu(), labels))

            _,_,_,_,sens,spec,f1, prec = confusion_matrix(outputs.cpu(), labels, 5, cur_batch_size)
            train_sensitivity.update(sens)
            train_specificity.update(spec)
            train_f1_score.update(f1)
            train_precision.update(prec)
            train_gmean.update(g_mean(sens, spec))
            train_kappa.update(kappa(outputs.cpu(), labels))
            
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.is_neptune:
                run['train/epoch/batch_loss'].log(losses.val)   
                run['train/epoch/batch_accuracy'].log(train_accuracy.val)
                run['epoch'].log(epoch_idx)

            if batch_idx % 100 == 0:
                
                msg = 'Epoch: [{0}/{3}][{1}/{2}]\t' \
                    'Train_Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                    'Train_Acc {train_acc.val:.5f} ({train_acc.avg:.5f})\t'\
                    'Train_G-Mean {train_gmean.val:.5f}({train_gmean.avg:.5f})\t'\
                    'Train_Kappa {train_kap.val:.5f}({train_kap.avg:.5f})\t'\
                    'Train_MF1 {train_mf1.val:.5f}({train_mf1.avg:.5f})\t'\
                    'Train_Precision {train_prec.val:.5f}({train_prec.avg:.5f})\t'\
                    'Train_Sensitivity {train_sens.val:.5f}({train_sens.avg:.5f})\t'\
                    'Train_Specificity {train_spec.val:.5f}({train_spec.avg:.5f})\t'\
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                        epoch_idx+1, batch_idx, len(train_data_loader),args.n_epochs, batch_time=batch_time,
                        speed=data_input[0].size(0)/batch_time.val,
                        data_time=data_time, loss=losses, train_acc = train_accuracy,
                        train_sens =train_sensitivity, train_spec = train_specificity, train_gmean = train_gmean,
                        train_kap = train_kappa, train_mf1 = train_f1_score, train_prec = train_precision)
                print(msg)


        #evaluation
        with torch.no_grad():
            Net.eval()
            for batch_val_idx, data_val in enumerate(val_data_loader):
                val_psg, val_labels = data_val
                val_eeg = val_psg[:,:,0,:]
                val_eog = val_psg[:,:,1,:]
                val_emg = val_psg[:,:,2,:]
                cur_val_batch_size = len(val_eeg)

                if args.model == 'CMT':
                    pred,_,_= Net(val_eeg.float().to(device), val_eog.float().to(device), val_emg.float().to(device))
                if args.model == 'USleep':
                    pred,_ = Net(val_psg.float().to(device))
                
                val_loss = criterion(pred.cpu(), val_labels)#.to(device))
                val_losses.update(val_loss.data.item())
                val_accuracy.update(accuracy(pred.cpu(), val_labels))

                sens_list,spec_list,f1_list,prec_list, sens,spec,f1,prec = confusion_matrix(pred.cpu(), val_labels,  5, cur_val_batch_size)
                val_sensitivity.update(sens)
                val_specificity.update(spec)
                val_f1_score.update(f1)
                val_precision.update(prec)
                val_gmean.update(g_mean(sens, spec))
                val_kappa.update(kappa(pred.cpu(), val_labels))

                class1_sens.update(sens_list[0])
                class2_sens.update(sens_list[1])
                class3_sens.update(sens_list[2])
                class4_sens.update(sens_list[3])
                class5_sens.update(sens_list[4])

                class1_spec.update(spec_list[0])
                class2_spec.update(spec_list[1])
                class3_spec.update(spec_list[2])
                class4_spec.update(spec_list[3])
                class5_spec.update(spec_list[4])

                class1_f1.update(f1_list[0])
                class2_f1.update(f1_list[1])
                class3_f1.update(f1_list[2])
                class4_f1.update(f1_list[3])
                class5_f1.update(f1_list[4])

            print(batch_val_idx)

        

            print(f'===========================================================Epoch : [{epoch_idx+1}/{args.n_epochs}]  Evaluation ===========================================================================================================>')
            print("Training Results : ")
            print(f"Training Loss     : {losses.avg}, Training Accuracy      : {train_accuracy.avg}, Training G-Mean      : {train_gmean.avg}") 
            print(f"Training Kappa      : {train_kappa.avg},Training MF1     : {train_f1_score.avg}, Training Precision      : {train_precision.avg}, Training Sensitivity      : {train_sensitivity.avg}, Training Specificity      : {train_specificity.avg}")
        
            print("Validation Results : ")
            print(f"Validation Loss   : {val_losses.avg}, Validation Accuracy : {val_accuracy.avg}, Validation G-Mean      : {val_gmean.avg}") 
            print(f"Validation Kappa     : {val_kappa.avg}, Validation MF1      : {val_f1_score.avg}, Validation Precision      : {val_precision.avg},  Validation Sensitivity      : {val_sensitivity.avg}, Validation Specificity      : {val_specificity.avg}")
        

            print(f"Class wise sensitivity W: {class1_sens.avg}, S1: {class2_sens.avg}, S2: {class3_sens.avg}, S3: {class4_sens.avg}, R: {class5_sens.avg}")
            print(f"Class wise specificity W: {class1_spec.avg}, S1: {class2_spec.avg}, S2: {class3_spec.avg}, S3: {class4_spec.avg}, R: {class5_spec.avg}")
            print(f"Class wise F1  W: {class1_f1.avg}, S1: {class2_f1.avg}, S2: {class3_f1.avg}, S3: {class4_f1.avg}, R: {class5_f1.avg}")

            if args.is_neptune:
                run['train/epoch/epoch_train_loss'].log(losses.avg)
                run['train/epoch/epoch_val_loss'].log(val_losses.avg)

                run['train/epoch/epoch_train_accuracy'].log(train_accuracy.avg)
                run['train/epoch/epoch_val_accuracy'].log(val_accuracy.avg)

                run['train/epoch/epoch_train_sensitivity'].log(train_sensitivity.avg)
                run['train/epoch/epoch_val_sensitivity'].log(val_sensitivity.avg)

                run['train/epoch/epoch_train_specificity'].log(train_specificity.avg)
                run['train/epoch/epoch_val_specificity'].log(val_specificity.avg)

                run['train/epoch/epoch_train_G-Mean'].log(train_gmean.avg)
                run['train/epoch/epoch_val_G-Mean'].log(val_gmean.avg)

                run['train/epoch/epoch_train_Kappa'].log(train_kappa.avg)
                run['train/epoch/epoch_val_Kappa'].log(val_kappa.avg)

                run['train/epoch/epoch_train_MF1 Score'].log(train_f1_score.avg)
                run['train/epoch/epoch_val_MF1 Score'].log(val_f1_score.avg)

                run['train/epoch/epoch_train_Precision'].log(train_precision.avg)
                run['train/epoch/epoch_val_Precision'].log(val_precision.avg)

      #################################
      
                run['train/epoch/epoch_val_Class wise sensitivity W'].log(class1_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity S1'].log(class2_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity S2'].log(class3_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity S3'].log(class4_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity R'].log(class5_sens.avg)

                run['train/epoch/epoch_val_Class wise specificity W'].log(class1_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity S1'].log(class2_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity S2'].log(class3_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity S3'].log(class4_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity R'].log(class5_spec.avg)

                run['train/epoch/epoch_val_Class wise F1 Score W'].log(class1_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score S1'].log(class2_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score S2'].log(class3_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score S3'].log(class4_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score R'].log(class5_f1.avg)

            if val_accuracy.avg > best_val_acc or (epoch_idx+1)% 50==0 or val_kappa.avg > best_val_kappa:
                if val_accuracy.avg > best_val_acc:
                
                    best_val_acc = val_accuracy.avg
                    print("================================================================================================")
                    print("                                          Saving Best Model (ACC)                                     ")
                    print("================================================================================================")
                    torch.save(Net, f'{args.project_path}/model_check_points/checkpoint_model_epoch_best_acc.pth.tar')

                if val_kappa.avg > best_val_kappa:
                
                    best_val_kappa = val_kappa.avg
                    print("================================================================================================")
                    print("                                          Saving Best Model (Kappa)                                    ")
                    print("================================================================================================")
                    torch.save(Net, f'{args.project_path}/model_check_points/checkpoint_model_epoch_best_kappa.pth.tar')

                if (epoch_idx+1)% 50==0 :
                    torch.save(Net, f'{args.project_path}/model_check_points//checkpoint_model_epoch_last.pth.tar')

                if args.is_neptune:
                    run['model/best_acc'].log(val_accuracy.avg)
                    run['model/best_kappa'].log(val_kappa.avg)
    print('========================================Finished Training ===========================================')

    


def KD_online_training(Net_s,Net_t,train_data_loader,labels1,val_data_loader,labels2,criterion_ce,criterion_kl,criterion_mse,optimizer_t, optimizer_s,args,device):
    # Training the model
    
    torch.autograd.set_detect_anomaly(True)
    if args.is_neptune:
        run = neptune.init(project= "mithunjha/earEEG-v2-cross", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZjA0YTVhOC02ZGVlLTQ0NTktOWY3NS03YzFhZWUxY2M4MTcifQ==")
        parameters = {"Experiment" : "KD online training",
        'Loss': "Categorical Crossentropy + MSE Loss",
        'Model Type' : args.model,
        'depth' : args.depth,
        'd_model' : args.d_model,
        'dim_feedforward' : args.dim_feedforward, 
        'window_size ':args.window_size ,
        'Batch Size': args.batch_size,
        'Optimizer' : "Adam",
        'Learning Rate': args.lr,
        'eps' : args.eps,
        'Beta 1': args.beta_1,
        'Beta 2': args.beta_2,
        'n_epochs': args.n_epochs,
        'val_set' : args.val_data_list[0]+1}
        run['model/parameters'] = parameters
        run['model/model_architecture'] = Net_t

    best_val_acc = 0
    best_val_kappa = 0

    train_losses_s = []
    train_losses_t = []
    val_losses_s = []
    val_losses_t = []
    train_accuracy_s = []
    train_accuracy_t = []
    val_accuracy_s = []
    val_accuracy_t = []
    for epoch_idx in range(args.n_epochs):  # loop over the dataset multiple times
        #被试划分
        mdd_train_index = [7, 15, 2, 9, 12, 4, 17, 1, 10, 6, 8, 3, 14, 5]
        mdd_val_index = [13, 11, 0, 16]

        hc_train_index = [20, 27, 28, 31, 19, 35, 38, 23, 30, 36, 34, 18, 37, 21]
        hc_val_index =  [22, 25, 26, 29]

        train_index = mdd_train_index + hc_train_index
        val_index = mdd_val_index + hc_val_index

        # # 确保模型参数需要梯度
        # for param in Net_t.parameters():
        #     param.requires_grad = True

        # for param in Net_s.parameters():
        #     param.requires_grad = True

        Net_s.train()
        Net_t.train()        
        print(f'===========================================================Training Epoch : [{epoch_idx+1}/{args.n_epochs}] ===========================================================================================================>')
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        losses_t = AverageMeter()
        losses_s = AverageMeter()
        val_t_losses = AverageMeter()
        val_s_losses = AverageMeter()
        
        train_t_accuracy = AverageMeter()
        train_s_accuracy = AverageMeter()
        val_t_accuracy = AverageMeter()
        val_s_accuracy = AverageMeter()

        train_t_kappa = AverageMeter()
        train_s_kappa = AverageMeter()
        val_t_kappa = AverageMeter()
        val_s_kappa = AverageMeter()

        end = time.time()

        for batch_idx, (data_input1, data_input2) in enumerate(zip(train_data_loader, val_data_loader)):
            print(batch_idx)
            data_time.update(time.time() - end)
            # 数据截取（前40段）
            data_input1 = data_input1[:,0:40,:,:]            
            labels1 = labels1[:,0:40,:,:]
            labels2 = labels2[:,0:40,:,:]
            
            # 数据加载
            train_input1 = data_input1[train_index].reshape(-1,128,378)
            train_input2 = data_input2[train_index].reshape(-1,3,253)
            #数据切分
            data_train_input1 = train_input1[:,:,:250] # x 128 250
            adj_train_input1 = train_input1[:,:,250:378] # x 128 128
            # 对每个通道的时间序列进行标准化（Z-score归一化）
            # 计算每个通道的均值和标准差
            # means = data_train_input1.mean(dim=2, keepdim=True)
            # stds = data_train_input1.std(dim=2, keepdim=True)
            # # 标准化
            # data_train_input1.sub_(means).div_(stds)
            # data_train_input1 = torch.tensor(data_train_input1, requires_grad=True)
            # adj_train_input1 = torch.tensor(adj_train_input1, requires_grad=True)
            data_train_input2 = train_input2[:,:,:250] # x 3 250
            adj_train_input2 = train_input2[:,:,250:253] # x 3 3
            # means2 = data_train_input2.mean(dim=2, keepdim=True)
            # stds2 = data_train_input2.std(dim=2, keepdim=True)
            # # 标准化
            # data_train_input2.sub_(means2).div_(stds2)
            # data_train_input2 = torch.tensor(data_train_input2, requires_grad=True)
            # 标签处理
            train_labels1_tmp = labels1[train_index].reshape(-1,128).astype(float)
            train_labels1_tmp2 = train_labels1_tmp[:,0]
            train_labels1 = torch.tensor(train_labels1_tmp2,dtype=torch.float32, requires_grad=True)

            train_labels2_tmp = labels2[train_index].reshape(-1,3).astype(float)
            train_labels2_tmp2 = train_labels2_tmp[:,0]
            train_labels2 = torch.tensor(train_labels2_tmp2,dtype=torch.float32, requires_grad=True)

            targets,_,_ = Net_t(data_train_input1.float().to(device),#.requires_grad_(True)
                                adj_train_input1.float().to(device))
            # targets,_,_ = checkpoint(Net_t,data_train_input1.float().to(device),adj_train_input1.float().to(device))
            # print(targets.shape)
            train_pred = F.softmax(targets,dim = 1).cpu()
            train_pred_ = F.softmax(targets,dim = 1).cpu().argmax(dim = 1)
            data_train_input2 = data_train_input2.unsqueeze(1) 
            # print(data_train_input2.shape) # torch.Size([1120, 1, 3, 250])
            # outputs = Net_s(data_train_input2.float().to(device))
            outputs = checkpoint(Net_s,data_train_input2.float().to(device).requires_grad_(True))
            # print(outputs.shape)
            train_pred2 = F.softmax(outputs,dim = 1).cpu()
            train_pred2_log = F.log_softmax(outputs,dim = 1).cpu()
            train_pred2_ = F.softmax(outputs,dim = 1).cpu().argmax(dim = 1)
            # print(pred.shape, pred2.shape)
            loss_t = criterion_ce(targets.to(dtype=torch.float32).cpu(),train_labels1.long()) 
            # print(pred2.shape, train_labels2.shape) #torch.Size([1120]) (1120,)
            # print(outputs.shape, targets.shape)# torch.Size([1120, 2]) torch.Size([2240, 2])
            loss_s =  criterion_ce(outputs.to(dtype=torch.float32).cpu(),train_labels2.long()) \
                    # .add(- criterion_kl(train_pred,-train_pred2_log)) \
                    # .add(criterion_mse(train_pred_.float(),train_pred2_.float()))
            # loss_s =  criterion_ce(outputs.to(dtype=torch.float32).cpu(),train_labels2.long())
            print("loss_t:",loss_t, "loss_s:",loss_s)
            #记录loss
            
            optimizer_s.zero_grad()
            loss_s.backward(retain_graph=True)#############
            optimizer_s.step()
            torch.cuda.empty_cache()  # 清理未使用的缓存内存
            optimizer_t.zero_grad()
            loss_t.backward()### retain_graph=True
            optimizer_t.step()
            
            losses_t.update(loss_t.data.item())
            losses_s.update(loss_s.data.item())

            train_losses_s.append(loss_s.detach().cpu().numpy())
            train_losses_t.append(loss_t.detach().cpu().numpy())

            train_t_accuracy.update(accuracy(targets.cpu(), train_labels1))
            train_s_accuracy.update(accuracy(outputs.cpu(), train_labels2))
            #记录acc
            train_accuracy_s.append(train_s_accuracy.val)
            train_accuracy_t.append(train_t_accuracy.val)
            train_t_kappa.update(kappa(targets.cpu(), train_labels1))
            train_s_kappa.update(kappa(outputs.cpu(), train_labels2))
            # print(outputs.shape, labels.shape)
            if args.is_neptune:
                run['train/epoch/batch_t_loss'].log(losses_t.val)     #1
                run['train/epoch/batch_s_loss'].log(losses_s.val) 
                run['train/epoch/batch_t_accuracy'].log(train_t_accuracy.val)
                run['train/epoch/batch_s_accuracy'].log(train_s_accuracy.val)
                run['epoch'].log(epoch_idx)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if batch_idx % 10 == 0:
                
                msg = 'Epoch: [{0}/{3}][{1}/{2}]\t' \
                    'Train_t_Loss {loss_t.val:.5f} ({loss_t.avg:.5f})\t'\
                    'Train_s_Loss {loss_s.val:.5f} ({loss_s.avg:.5f})\t'\
                    'Train_t_Acc {train_t_acc.val:.5f} ({train_t_acc.avg:.5f})\t'\
                    'Train_s_Acc {train_s_acc.val:.5f} ({train_s_acc.avg:.5f})\t'\
                    'Train_t_Kappa {train_t_kap.val:.5f}({train_t_kap.avg:.5f})\t'\
                    'Train_s_Kappa {train_s_kap.val:.5f}({train_s_kap.avg:.5f})\t'\
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                        epoch_idx+1, batch_idx, len(train_data_loader),args.n_epochs, batch_time=batch_time,
                        speed=data_input1[0].size(0)/batch_time.val,
                        data_time=data_time, loss_t=losses_t, loss_s=losses_s, train_t_acc = train_t_accuracy, train_s_acc = train_s_accuracy,
                        train_t_kap = train_t_kappa,train_s_kap = train_s_kappa)
                print(msg)
        torch.cuda.empty_cache()  # 清理未使用的缓存内存

        #evaluation
        with torch.no_grad():
            Net_s.eval()
            Net_t.eval()
            for batch_val_idx, (data_input1, data_input2) in enumerate(zip(train_data_loader, val_data_loader)):
                data_time.update(time.time() - end)
                data_input1 = data_input1[:,0:40,:,:]
                val_input1 = data_input1[val_index].reshape(-1,128,378)
                val_input2 = data_input2[val_index].reshape(-1,3,253)

                data_val_input1 = val_input1[:,:,:250] # x 128 250
                adj_val_input1 = val_input1[:,:,250:378] # x 128 128
                # means1 = data_val_input1.mean(dim=2, keepdim=True)
                # stds1 = data_val_input1.std(dim=2, keepdim=True)
                # # 标准化
                # data_val_input1.sub_(means1).div_(stds1)
                # data_val_input1 = torch.tensor(data_val_input1, requires_grad=True)
                # adj_val_input1 = torch.tensor(adj_val_input1, requires_grad=True)

                data_val_input2 = val_input2[:,:,:250] # x 3 250
                adj_val_input2 = val_input2[:,:,250:253] # x 3 3
                # means2 = data_val_input2.mean(dim=2, keepdim=True)
                # stds2 = data_val_input2.std(dim=2, keepdim=True)
                # # 标准化
                # data_val_input2.sub_(means2).div_(stds2)
                data_val_input2 = data_val_input2.unsqueeze(1)
                # data_val_input2 = torch.tensor(data_val_input2, requires_grad=True)

                val_labels1_tmp = labels1[val_index].reshape(-1,128).astype(float)
                val_labels1_tmp = val_labels1_tmp[:,0]
                val_labels1 = torch.tensor(val_labels1_tmp,dtype=torch.float32)

                val_labels2_tmp = labels2[val_index].reshape(-1,3).astype(float)
                val_labels2_tmp = val_labels2_tmp[:,0]
                val_labels2 = torch.tensor(val_labels2_tmp ,dtype=torch.float32)

                # if args.model == 'CMT':
                #     val_targets,cls_val_t,_ = Net_t(val_psg[:,:,3,:].float().to(device), val_psg[:,:,4,:].float().to(device), val_psg[:,:,5,:].float().to(device),finetune = True)
                #     pred,cls_val_s,_ = Net_s(val_sig1.float().to(device), val_sig2.float().to(device), val_sig3.float().to(device),finetune = True)
                # if args.model == 'USleep':
                #     val_targets,cls_val_t = Net_t(val_psg[:,:,3:6,:].float().to(device))
                #     pred,cls_val_s = Net_s(val_psg[:,:,0:3,:].float().to(device))
                # val_targets,_,_ = Net_t(data_val_input1.float().to(device),adj_val_input1.float().to(device))
                val_targets,_,_ = checkpoint(Net_t,data_val_input1.float().to(device).requires_grad_(True),adj_val_input1.float().to(device).requires_grad_(True))
                pred = torch.softmax(val_targets,dim = 1).cpu()
                pred_ = pred.argmax(dim = 1)
                # print(pred.shape,pred.dtype)
                # val_outputs = Net_s(data_val_input2.float().to(device))
                val_outputs = checkpoint(Net_s,data_val_input2.float().to(device).requires_grad_(True))
                pred2 = torch.softmax(val_outputs,dim = 1).cpu()
                pred2_ = pred2.argmax(dim = 1)
                pred2_log = F.log_softmax(val_outputs,dim = 1).cpu()
                # print(val_targets.dtype, val_labels1.dtype) 
                val_t_loss = criterion_ce(val_targets.to(dtype=torch.float32).cpu(),val_labels1.long())
                val_s_loss = criterion_ce(val_outputs.cpu(),val_labels2.long()) 
                        #  -criterion_kl(pred, -pred2_log)\
                        # .add(criterion_mse(pred_.float(),pred2_.float())) 
                # val_s_loss = criterion_ce(val_outputs.to(dtype=torch.float32).cpu(),val_labels2.long())##???
                print("val_t_loss",val_t_loss, "val_s_loss",val_s_loss)

                val_t_losses.update(val_t_loss.data.item())
                val_s_losses.update(val_s_loss.data.item())
                
                val_losses_s.append(val_s_loss.detach().cpu().numpy())
                val_losses_t.append(val_t_loss.detach().cpu().numpy())
                val_t_accuracy.update(accuracy(val_targets.cpu(), val_labels1))
                val_s_accuracy.update(accuracy(val_outputs.cpu(), val_labels2))

                val_accuracy_s.append(val_s_accuracy.val)
                val_accuracy_t.append(val_t_accuracy.val)

                val_t_kappa.update(kappa(val_targets.cpu(), val_labels1))
                val_s_kappa.update(kappa(val_outputs.cpu(), val_labels2))

            
            print(f'===========================================================Epoch : [{epoch_idx+1}/{args.n_epochs}]  Evaluation ===========================================================================================================>')
            print("Training Results : ")
            print(f"Training T Loss     : {losses_t.avg}, Training T Accuracy      : {train_t_accuracy.avg}, Training T Kappa      : {train_t_kappa.avg}")
            print(f"Training S Loss     : {losses_s.avg}, Training S Accuracy      : {train_s_accuracy.avg}, Training S Kappa      : {train_s_kappa.avg}")
            print("Validation Results : ")
            print(f"Validation T Loss   : {val_t_losses.avg}, Validation T Accuracy : {val_t_accuracy.avg}, Validation T Kappa     : {val_t_kappa.avg}")
            print(f"Validation S Loss   : {val_s_losses.avg}, Validation S Accuracy : {val_s_accuracy.avg}, Validation S Kappa     : {val_s_kappa.avg}")

        
            if args.is_neptune:
                run['train/epoch/epoch_train_t_loss'].log(losses_t.avg)
                run['train/epoch/epoch_train_s_loss'].log(losses_s.avg)
                run['train/epoch/epoch_val_t_loss'].log(val_t_losses.avg)
                run['train/epoch/epoch_val_s_loss'].log(val_s_losses.avg)

                run['train/epoch/epoch_train_t_accuracy'].log(train_t_accuracy.avg)
                run['train/epoch/epoch_train_s_accuracy'].log(train_s_accuracy.avg)
                run['train/epoch/epoch_val_t_accuracy'].log(val_t_accuracy.avg)
                run['train/epoch/epoch_val_s_accuracy'].log(val_s_accuracy.avg)

                run['train/epoch/epoch_train_t_Kappa'].log(train_t_kappa.avg)
                run['train/epoch/epoch_train_s_Kappa'].log(train_s_kappa.avg)
                run['train/epoch/epoch_val_t_Kappa'].log(val_t_kappa.avg)
                run['train/epoch/epoch_val_s_Kappa'].log(val_s_kappa.avg)

            #if val_accuracy.avg > best_val_acc or (epoch_idx+1)%10==0 or val_kappa.avg > best_val_kappa:
            if val_s_accuracy.avg > best_val_acc or val_s_kappa.avg > best_val_kappa:
                if val_s_accuracy.avg > best_val_acc:
                    if args.is_neptune:
                        run['model/bestmodel_acc'].log(epoch_idx+1)
                    best_val_acc = val_s_accuracy.avg
                    print("================================================================================================")
                    print("                                          Saving Best Model (ACC)                                     ")
                    print("================================================================================================")
                    torch.save(Net_t, f'{args.project_path}/model_check_points/teacher_checkpoint_acc.pth.tar')
                    torch.save(Net_s, f'{args.project_path}/model_check_points/student_checkpoint_acc.pth.tar')

                if val_s_kappa.avg > best_val_kappa:
                    if args.is_neptune:
                        run['model/bestmodel_kappa'].log(epoch_idx+1)
                    best_val_kappa = val_s_kappa.avg
                    print("================================================================================================")
                    print("                                          Saving Best Model (Kappa)                                    ")
                    print("================================================================================================")
                    torch.save(Net_t, f'{args.project_path}/model_check_points/teacher_checkpoint_kappa.pth.tar')
                    torch.save(Net_s, f'{args.project_path}/model_check_points/student_checkpoint_kappa.pth.tar')

                if args.is_neptune:
                    run['model/best_acc'].log(val_s_accuracy.avg)
                    run['model/best_kappa'].log(val_s_kappa.avg)
        torch.cuda.empty_cache()

    print('========================================Finished Training ===========================================')
    epochs = range(1, args.n_epochs + 1)  # 假设有100个epoch

    # 创建一个figure对象，并设置大小
    plt.figure(figsize=(10, 6))

    # 绘制训练损失_s
    plt.plot(epochs, train_losses_s, label='Train Loss (s)', linestyle='-', color='blue')

    # 绘制训练损失_t
    plt.plot(epochs, train_losses_t, label='Train Loss (t)', linestyle='--', color='green')

    # 绘制验证损失_s
    plt.plot(epochs, val_losses_s, label='Validation Loss (s)', linestyle='-', color='red')

    # 绘制验证损失_t
    plt.plot(epochs, val_losses_t, label='Validation Loss (t)', linestyle='--', color='orange')

    # 添加标题和标签
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  # 添加图例

    # 显示图表
    plt.grid(True)  # 添加网格线
    plt.tight_layout()  # 自动调整布局
    plt.show()

    # figure2
    plt.figure(figsize=(10, 6))

    # 绘制训练损失_s
    plt.plot(epochs, train_accuracy_s, label='Train Accuracy (s)', linestyle='-', color='blue')

    # 绘制训练损失_t
    plt.plot(epochs, train_accuracy_t, label='Train Accuracy (t)', linestyle='--', color='green')

    # 绘制验证损失_s
    plt.plot(epochs, val_accuracy_s, label='Validation Accuracy (s)', linestyle='-', color='red')

    # 绘制验证损失_t
    plt.plot(epochs, val_accuracy_t, label='Validation Accuracy (t)', linestyle='--', color='orange')

    # 添加标题和标签
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()  # 添加图例

    # 显示图表
    plt.grid(True)  # 添加网格线
    plt.tight_layout()  # 自动调整布局
    plt.show()


def KD_offline_training(Net_s,Net_t,train_data_loader,val_data_loader,criterion_ce,criterion_mse,optimizer_s,args,device):
    # Training the model
    warnings.filterwarnings("ignore")
    if args.is_neptune:
        run = neptune.init(project= "mithunjha/earEEG-v2-cross", api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwZjA0YTVhOC02ZGVlLTQ0NTktOWY3NS03YzFhZWUxY2M4MTcifQ==")
        parameters = {"Experiment" : "KD Offline training", 
        'Loss': "Categorical Crossentropy + MSE Loss",
        'Model Type' : args.model,
        'depth' : args.depth,
        'd_model' : args.d_model,
        'dim_feedforward' : args.dim_feedforward, 
        'window_size ':args.window_size ,
        'Batch Size': args.batch_size,
        'Optimizer' : "Adam",
        'Learning Rate': args.lr,
        'eps' : args.eps,
        'Beta 1': args.beta_1,
        'Beta 2': args.beta_2,
        'n_epochs': args.n_epochs,
        'val_set' : args.val_data_list[0]+1}
        run['model/parameters'] = parameters
        run['model/model_architecture'] = Net_t
    best_val_acc = 0
    best_val_kappa = 0

    for epoch_idx in range(args.n_epochs):  # loop over the dataset multiple times
        
        Net_s.train()
        Net_t.eval()        ### Check whether weights of the teacher gets updated
        print(f'===========================================================Training Epoch : [{epoch_idx+1}/{args.n_epochs}] ===========================================================================================================>')
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        losses = AverageMeter()
        val_losses = AverageMeter()
        val_kd_loss = AverageMeter()
        
        train_accuracy = AverageMeter()
        val_accuracy = AverageMeter()
        val_teach_accuracy = AverageMeter()

        train_sensitivity = AverageMeter()
        val_sensitivity = AverageMeter()
        
        train_specificity = AverageMeter()
        val_specificity = AverageMeter()

        train_gmean = AverageMeter()
        val_gmean = AverageMeter()

        train_kappa = AverageMeter()
        val_kappa = AverageMeter()

        train_f1_score = AverageMeter()
        val_f1_score = AverageMeter()

        train_precision = AverageMeter()
        val_precision = AverageMeter()

        class1_sens = AverageMeter()
        class2_sens = AverageMeter()
        class3_sens = AverageMeter()
        class4_sens = AverageMeter()
        class5_sens = AverageMeter()

        class1_spec = AverageMeter()
        class2_spec = AverageMeter()
        class3_spec = AverageMeter()
        class4_spec = AverageMeter()
        class5_spec = AverageMeter()

        class1_f1 = AverageMeter()
        class2_f1 = AverageMeter()
        class3_f1 = AverageMeter()
        class4_f1 = AverageMeter()
        class5_f1 = AverageMeter()

        end = time.time()

        for batch_idx, data_input in enumerate(train_data_loader):
            # get the inputs; data is a list of [inputs, labels]
            data_time.update(time.time() - end)
            psg, labels = data_input
            sig1 = psg[:,:,0,:]# L-R
            sig2 = psg[:,:,1,:]# L
            sig3 = psg[:,:,2,:]# R
            sig4 = psg[:,:,3,:]# c3-01
            sig5 = psg[:,:,4,:]# c4-o2
            sig6 = psg[:,:,5,:]# eog
            cur_batch_size = len(sig1)
            
            # zero the parameter gradients
            

            with torch.no_grad():
                if args.model == 'CMT':
                    targets,cls_t_feat,_ = Net_t(sig4.float().to(device), sig5.float().to(device), sig6.float().to(device),finetune = True)
                if args.model == 'USleep':
                    targets,cls_t_feat = Net_t(psg[:,:,3:6,:].float().to(device))
            optimizer_s.zero_grad()

            if args.model == 'CMT':
                outputs,cls_s_feat,_ = Net_s(sig1.float().to(device), sig2.float().to(device), sig3.float().to(device),finetune = True)
            if args.model == 'USleep':
                outputs,cls_s_feat = Net_s(psg[:,:,0:3,:].float().to(device))

            
            loss =  criterion_ce(outputs.cpu(),labels) + criterion_mse(cls_s_feat,cls_t_feat.detach())
            

            loss.backward()
            optimizer_s.step()
            losses.update(loss.data.item())
            train_accuracy.update(accuracy(outputs.cpu(), labels))

            _,_,_,_,sens,spec,f1, prec = confusion_matrix(outputs.cpu(), labels, 5, cur_batch_size)
            train_sensitivity.update(sens)
            train_specificity.update(spec)
            train_f1_score.update(f1)
            train_precision.update(prec)
            train_gmean.update(g_mean(sens, spec))
            train_kappa.update(kappa(outputs.cpu(), labels))
            # print(outputs.shape, labels.shape)

            if args.is_neptune:
                run['train/epoch/batch_loss'].log(losses.val)     #1
                run['train/epoch/batch_accuracy'].log(train_accuracy.val)
                run['epoch'].log(epoch_idx)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if batch_idx % 100 == 0:
                
                msg = 'Epoch: [{0}/{3}][{1}/{2}]\t' \
                    'Train_Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                    'Train_Acc {train_acc.val:.5f} ({train_acc.avg:.5f})\t'\
                    'Train_G-Mean {train_gmean.val:.5f}({train_gmean.avg:.5f})\t'\
                    'Train_Kappa {train_kap.val:.5f}({train_kap.avg:.5f})\t'\
                    'Train_MF1 {train_mf1.val:.5f}({train_mf1.avg:.5f})\t'\
                    'Train_Precision {train_prec.val:.5f}({train_prec.avg:.5f})\t'\
                    'Train_Sensitivity {train_sens.val:.5f}({train_sens.avg:.5f})\t'\
                    'Train_Specificity {train_spec.val:.5f}({train_spec.avg:.5f})\t'\
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                        epoch_idx+1, batch_idx, len(train_data_loader),args.n_epochs, batch_time=batch_time,
                        speed=data_input[0].size(0)/batch_time.val,
                        data_time=data_time, loss=losses, train_acc = train_accuracy,
                        train_sens =train_sensitivity, train_spec = train_specificity, train_gmean = train_gmean,
                        train_kap = train_kappa, train_mf1 = train_f1_score, train_prec = train_precision)
                print(msg)


        #evaluation
        with torch.no_grad():
            Net_s.eval()
            Net_t.eval()
            for batch_val_idx, data_val in enumerate(val_data_loader):
                val_psg, val_labels = data_val
                val_sig1 = val_psg[:,:,0,:]
                val_sig2 = val_psg[:,:,1,:]
                val_sig3 = val_psg[:,:,2,:]
                cur_val_batch_size = len(val_sig1)

                if args.model == 'CMT':
                    val_targets,cls_val_t,_ = Net_t(val_psg[:,:,3,:].float().to(device), val_psg[:,:,4,:].float().to(device), val_psg[:,:,5,:].float().to(device),finetune = True)
                    pred,cls_val_s,_ = Net_s(val_sig1.float().to(device), val_sig2.float().to(device), val_sig3.float().to(device),finetune = True)
                if args.model == 'USleep':
                    val_targets,cls_val_t = Net_t(val_psg[:,:,3:6,:].float().to(device))
                    pred,cls_val_s = Net_s(val_psg[:,:,0:3,:].float().to(device))


                val_loss = criterion_ce(pred.cpu(), val_labels) + + criterion_mse(cls_val_s,cls_val_t.detach())
                val_losses.update(val_loss.data.item())
                val_accuracy.update(accuracy(pred.cpu(), val_labels))

                sens_list,spec_list,f1_list,prec_list, sens,spec,f1,prec = confusion_matrix(pred.cpu(), val_labels,  5, cur_val_batch_size)
                val_sensitivity.update(sens)
                val_specificity.update(spec)
                val_f1_score.update(f1)
                val_precision.update(prec)
                val_gmean.update(g_mean(sens, spec))
                val_kappa.update(kappa(pred.cpu(), val_labels))



                class1_sens.update(sens_list[0])
                class2_sens.update(sens_list[1])
                class3_sens.update(sens_list[2])
                class4_sens.update(sens_list[3])
                class5_sens.update(sens_list[4])

                class1_spec.update(spec_list[0])
                class2_spec.update(spec_list[1])
                class3_spec.update(spec_list[2])
                class4_spec.update(spec_list[3])
                class5_spec.update(spec_list[4])

                class1_f1.update(f1_list[0])
                class2_f1.update(f1_list[1])
                class3_f1.update(f1_list[2])
                class4_f1.update(f1_list[3])
                class5_f1.update(f1_list[4])

                val_targets,_,_ = Net_t(val_psg[:,:,3,:].float().to(device), val_psg[:,:,4,:].float().to(device), val_psg[:,:,5,:].float().to(device),finetune = True)
                val_teach_accuracy.update(accuracy(val_targets.cpu(), val_labels))
                val_kd_loss.update(criterion_mse(pred.cpu(),val_targets.cpu()).data.item())
            print(batch_val_idx)

            

            print(f'===========================================================Epoch : [{epoch_idx+1}/{args.n_epochs}]  Evaluation ===========================================================================================================>')
            print("Training Results : ")
            print(f"Training Loss     : {losses.avg}, Training Accuracy      : {train_accuracy.avg}, Training G-Mean      : {train_gmean.avg}") 
            print(f"Training Kappa      : {train_kappa.avg},Training MF1     : {train_f1_score.avg}, Training Precision      : {train_precision.avg}, Training Sensitivity      : {train_sensitivity.avg}, Training Specificity      : {train_specificity.avg}")
            
            print("Validation Results : ")
            print(f"Validation Loss   : {val_losses.avg}, Validation Accuracy : {val_accuracy.avg}, Validation G-Mean      : {val_gmean.avg}") 
            print(f"Validation Kappa     : {val_kappa.avg}, Validation MF1      : {val_f1_score.avg}, Validation Precision      : {val_precision.avg},  Validation Sensitivity      : {val_sensitivity.avg}, Validation Specificity      : {val_specificity.avg}")
            print(f"Validation T Acc     : {val_teach_accuracy.avg}, Val_KD_Loss :{val_kd_loss.avg}")

            print(f"Class wise sensitivity W: {class1_sens.avg}, S1: {class2_sens.avg}, S2: {class3_sens.avg}, S3: {class4_sens.avg}, R: {class5_sens.avg}")
            print(f"Class wise specificity W: {class1_spec.avg}, S1: {class2_spec.avg}, S2: {class3_spec.avg}, S3: {class4_spec.avg}, R: {class5_spec.avg}")
            print(f"Class wise F1  W: {class1_f1.avg}, S1: {class2_f1.avg}, S2: {class3_f1.avg}, S3: {class4_f1.avg}, R: {class5_f1.avg}")

            if args.is_neptune:
                run['train/epoch/epoch_train_loss'].log(losses.avg)
                run['train/epoch/epoch_val_loss'].log(val_losses.avg)
                run['train/epoch/epoch_val_kd_loss'].log(val_kd_loss.avg)

                run['train/epoch/epoch_train_accuracy'].log(train_accuracy.avg)
                run['train/epoch/epoch_val_accuracy'].log(val_accuracy.avg)
                run['train/epoch/epoch_val_teach_accuracy'].log(val_teach_accuracy.avg)

                run['train/epoch/epoch_train_sensitivity'].log(train_sensitivity.avg)
                run['train/epoch/epoch_val_sensitivity'].log(val_sensitivity.avg)

                run['train/epoch/epoch_train_specificity'].log(train_specificity.avg)
                run['train/epoch/epoch_val_specificity'].log(val_specificity.avg)

                run['train/epoch/epoch_train_G-Mean'].log(train_gmean.avg)
                run['train/epoch/epoch_val_G-Mean'].log(val_gmean.avg)

                run['train/epoch/epoch_train_Kappa'].log(train_kappa.avg)
                run['train/epoch/epoch_val_Kappa'].log(val_kappa.avg)

                run['train/epoch/epoch_train_MF1 Score'].log(train_f1_score.avg)
                run['train/epoch/epoch_val_MF1 Score'].log(val_f1_score.avg)

                run['train/epoch/epoch_train_Precision'].log(train_precision.avg)
                run['train/epoch/epoch_val_Precision'].log(val_precision.avg)

                #################################
                
                run['train/epoch/epoch_val_Class wise sensitivity W'].log(class1_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity S1'].log(class2_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity S2'].log(class3_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity S3'].log(class4_sens.avg)
                run['train/epoch/epoch_val_Class wise sensitivity R'].log(class5_sens.avg)

                run['train/epoch/epoch_val_Class wise specificity W'].log(class1_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity S1'].log(class2_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity S2'].log(class3_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity S3'].log(class4_spec.avg)
                run['train/epoch/epoch_val_Class wise specificity R'].log(class5_spec.avg)

                run['train/epoch/epoch_val_Class wise F1 Score W'].log(class1_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score S1'].log(class2_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score S2'].log(class3_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score S3'].log(class4_f1.avg)
                run['train/epoch/epoch_val_Class wise F1 Score R'].log(class5_f1.avg)

            #if val_accuracy.avg > best_val_acc or (epoch_idx+1)%10==0 or val_kappa.avg > best_val_kappa:
            if val_accuracy.avg > best_val_acc or val_kappa.avg > best_val_kappa:
                if val_accuracy.avg > best_val_acc:
                    if args.is_neptune:
                        run['model/bestmodel_acc'].log(epoch_idx+1)
                    best_val_acc = val_accuracy.avg
                    print("================================================================================================")
                    print("                                          Saving Best Model (ACC)                                     ")
                    print("================================================================================================")
                    torch.save(Net_s, f'{args.project_path}/model_check_points/student_checkpoint_acc.pth.tar')

                if val_kappa.avg > best_val_kappa:
                    if args.is_neptune:
                        run['model/bestmodel_kappa'].log(epoch_idx+1)
                    best_val_kappa = val_kappa.avg
                    print("================================================================================================")
                    print("                                          Saving Best Model (Kappa)                                    ")
                    print("================================================================================================")
                    torch.save(Net_s, f'{args.project_path}/model_check_points/student_checkpoint_kappa.pth.tar')
                if args.is_neptune:
                    run['model/best_acc'].log(val_accuracy.avg)
                    run['model/best_kappa'].log(val_kappa.avg)
              
    print('========================================Finished Training ===========================================')