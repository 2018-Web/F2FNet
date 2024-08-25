import torch
import torch.nn as nn
import argparse
import os
from Datasets.Dataset_loader import get_dataset,EEG_Dataset
from Model.CrossModalTransformer import Cross_Transformer_Network
from Model.USleep import USleep, _EncoderBlock, _DecoderBlock
from Model.DGCNN import DGCNN
from Model.GNN import NET_MODEL
from Model.EEGNet import EEGNet
from Model.Training import supervised_training,KD_online_training,KD_offline_training
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
def parse_option():
    parser = argparse.ArgumentParser('Argument for training')
    parser.add_argument('--project_path', type=str, default='./results', help='Path to store project results')
    parser.add_argument('--data_path', type=str, default='',help='Path to the dataset file')
    parser.add_argument('--train_data_list', nargs="+", default = ['[0,1,2,3,5,6,7]'] ,  help='Folds in the dataset for training')
    parser.add_argument('--val_data_list', nargs="+", default = ['[8]'] ,  help='Folds in the dataset for validation')
    parser.add_argument('--is_retrain', type=bool, default=False,   help='To retrain a from saved checkpoint')
    parser.add_argument('--is_student_pretrain', type=bool, default=False,   help='To used pretrained model for student')
    parser.add_argument('--is_teacher_pretrain', type=bool, default=False,   help='To used pretrained model for teacher (online KD only')
    parser.add_argument('--model_path', type=str, default="",   help='Path to saved checkpoint for retraining')
    parser.add_argument('--student_model_path', type=str, default="gnn",   help='Path to saved checkpoint for student initialization')
    parser.add_argument('--signals', type=str, default = 'ear-eeg'  ,choices=['ear-eeg', 'scalp-eeg'],  help='signal type')
    parser.add_argument('--model', type=str, default = 'GCN'  ,choices=['GCN', 'DGCNN'],  help='Model architecture')
    #model parameters
    parser.add_argument('--training_type', type=str, default = 'Knowledge_distillation'  ,choices=['supervised', 'Knowledge_distillation'],  help='training type')
    parser.add_argument('--KD_type', type=str, default = 'online'  ,choices=['offline', 'online'],  help='Knowledge distillation type')
    parser.add_argument('--d_model', type=int, default = 256,  help='Embedding size of the CMT')
    parser.add_argument('--dim_feedforward', type=int, default = 1024,  help='No of neurons feed forward block')
    parser.add_argument('--window_size', type=int, default = 50,  help='Size of non-overlapping window')
    parser.add_argument('--depth', type=int, default = 12,  help='depth of USleep model')
    #training parameters
    parser.add_argument('--batch_size', type=int, default = 32 ,  help='Batch Size')
    
    #For Optimizer
    parser.add_argument('--lr', type=float, default = 0.01 ,  help='Learning rate')
    parser.add_argument('--beta_1', type=float, default = 0.9 ,  help='beta 1 for adam optimizer')
    parser.add_argument('--beta_2', type=float, default = 0.999 ,  help='beta 2 for adam optimizer')
    parser.add_argument('--eps', type=float, default = 1e-9 ,  help='eps for adam optimizer')
    parser.add_argument('--weight_decay', type=float, default = 0.0001 ,  help='weight_decay  for adam optimizer')
    parser.add_argument('--n_epochs', type=int, default = 200 ,  help='No of training epochs')
    

    #Neptune
    parser.add_argument('--is_neptune', type=bool, default=False, help='Is neptune used to track experiments')
    parser.add_argument('--nep_project', type=str, default='', help='Neptune Project Name')
    parser.add_argument('--nep_api', type=str, default='', help='Neptune API Token')
        
    opt = parser.parse_args()
    
    return opt

# experiment = 1

def main():
    args = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True
    print(device)
    print("<=============== Training arguments ===============>")
    for arg in vars(args):
        print(f"{arg} : {getattr(args,arg)}")
    
    if not os.path.isdir(args.project_path):
        os.makedirs(args.project_path)
        print(f"Project directory created at {args.project_path}")
    else:
        print(f"Project directory available at {args.project_path}")


    print("<=============== Getting Dataset ===============>")
    data_loader1,labels1,data_loader2,labels2,data_loader3,labels3 = get_dataset(args,device)
    # print(data_loader1)
    if not os.path.isdir(os.path.join(args.project_path,"model_check_points")):
        os.makedirs(os.path.join(args.project_path,"model_check_points"))

    if args.training_type == 'supervised':
        if args.is_retrain:
            #暂时未激活
            print(f"Loading previous model from {args.model_path}")
            Net = torch.load(f"{args.model_path}").to(device)
        else:
            print(f"Initializing Epoch XXX")
            if args.model == 'GCN':
                Net = NET_MODEL().to(device)
                # Net = DGCNN(in_channels=3, num_electrodes=128, hid_channels=32, num_layers=2, num_classes=2).to(device)
                # Net = Cross_Transformer_Network(d_model = args.d_model, dim_feedforward = args.dim_feedforward,window_size = args.window_size ).to(device)
            elif args.model == 'EEGNet':
                Net = EEGNet(chunk_size=250,num_electrodes=128,dropout=0.5,kernel_1=64,kernel_2=16,F1=8,F2=16,D=2,num_classes=2).to(device)
                # Net = USleep(in_chans=3,sfreq=200,depth=args.depth,with_skip_connection=True,n_classes=5,input_size_s=30,time_conv_size_s = 9/200,apply_softmax=False).to(device)
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(Net.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2),eps = args.eps, weight_decay = args.weight_decay)
        optimizer = torch.optim.Adam(Net.parameters(), lr=args.lr)
        supervised_training(Net, data_loader1,data_loader2, criterion, optimizer, args, device)
        #下一步在函数里
 
    if args.training_type == 'Knowledge_distillation':
        if args.KD_type == 'offline':
            #暂时未使用
            print(f"Loading previous model from {args.model_path} for teacher model")
            Net_t = torch.load(f"{args.model_path}").to(device)
            if args.is_student_pretrain:
                print(f"Loading previous model from {args.student_model_path} for student model")
                Net_s = torch.load(f"{args.student_model_path}").to(device)
            else:
                print(f"Initializing student model")
                if args.model == 'CMT':
                    Net_s = Cross_Transformer_Network(d_model = args.d_model, dim_feedforward = args.dim_feedforward,window_size = args.window_size ).to(device)
                elif args.model == 'USleep':
                    Net_s = USleep(in_chans=3,sfreq=200,depth=args.depth,with_skip_connection=True,n_classes=5,input_size_s=30,time_conv_size_s = 9/200,apply_softmax=False).to(device)

            criterion_mse =  nn.MSELoss()
            criterion_ce = nn.CrossEntropyLoss()
            optimizer_s = torch.optim.Adam(Net_s.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2),eps = args.eps)
            KD_offline_training(Net_s,Net_t,data_loader1,labels1,data_loader2,labels2,criterion_ce,criterion_mse,optimizer_s,args,device)
        
        elif args.KD_type == 'online': 
            if args.is_student_pretrain:# 预训练
                print(f"Loading previous model from {args.student_model_path} for student model")
                Net_s = torch.load(f"{args.student_model_path}").to(device)
                
            else:# 直接训练
                print(f"Initializing student model")
                # 选取学生模型
                # Net_s = DGCNN(in_channels=250, num_electrodes=3, hid_channels=32, num_layers=2, num_classes=2).to(device)
                Net_s = EEGNet(chunk_size=250,num_electrodes=3,dropout=0.5,kernel_1=64,kernel_2=16,F1=8,F2=16,D=2,num_classes=2).to(device)
                # if args.model == 'DGCNN':
                #     Net_s = DGCNN(in_channels=3, num_electrodes=3, hid_channels=32, num_layers=2, num_classes=2).to(device)
                # elif args.model == 'EEGNet':
                #     Net_s = EEGNet(chunk_size=250,num_electrodes=3,dropout=0.5,kernel_1=64,kernel_2=16,F1=8,F2=16,D=2,num_classes=2).to(device)

            if args.is_teacher_pretrain: # 预训练
                print(f"Loading previous model from {args.model_path} for teacher model")
                Net_t = torch.load(f"{args.model_path}").to(device)
            else: # 直接训练
                print(f"Initializing teacher model")
                # 选取教师模型
                if args.model == 'GCN':
                    Net_t = NET_MODEL().to(device)
                elif args.model == 'DGCNN':
                    Net_t = DGCNN(in_channels=250, num_electrodes=128, hid_channels=32, num_layers=2, num_classes=2).to(device)

            criterion_mse =  nn.MSELoss()
            criterion_ce = nn.CrossEntropyLoss() # Loss1
            # criterion_ce2 = # Loss2
            criterion_kl = nn.KLDivLoss(reduction='batchmean')# Loss3
            # 请注意，nn.KLDivLoss默认期望其输入是对数概率，如果你的模型输出的不是对数概率，你需要在传入损失函数之前对其进行对数转换。
            # criterion_l2 # Loss4 
            optimizer_s = torch.optim.Adam(Net_s.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2),eps = args.eps)
            optimizer_t = torch.optim.Adam(Net_t.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2),eps = args.eps)
            KD_online_training(Net_s,Net_t,data_loader1,labels1,data_loader2,labels2,criterion_ce,criterion_kl,criterion_mse,optimizer_t, optimizer_s,args,device)

    
if __name__ == '__main__':
    main()