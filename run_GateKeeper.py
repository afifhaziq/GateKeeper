
import time
import torch
import numpy as np
from train import train, init_network,test
import wandb

from importlib import import_module
import argparse
from utils_GateKeeper import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Encrypted Traffic Classification')
parser.add_argument('--test', type=bool, default=False, help='Train or Test.')
parser.add_argument('--data', type=str, required=True, help='input dataset source')



args = parser.parse_args()



def main():
    
    dataset = "C:\\Users\\afif\\Documents\\Master\\Code\\benchmark_ntc\\GateKeeper\\dataset\\" + args.data    
    model_name = 'GateKeeper' 
    
    x = import_module(model_name)

    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  


    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    
    train_iter = build_iterator(train_data, config)
    

    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    #print("args.data before calling get_time_dif:", args.data, type(args.data))
    pre_time_dif, average_pre_time = get_time_dif(start_time, test=0, data=args.data)
    print(f"Preprocess Time : {pre_time_dif:.10f} seconds")  # Show 6 decimal places
    print(f"Average prepocess time : {average_pre_time:.10f} seconds")
    
    
    
    model = x.Model(config).to(config.device)
    #init_network(model)
    
    #train(config, model, train_iter, dev_iter, test_iter)
    if args.test == False:
        print(args.test)
        print(model.parameters)
        train(config, model, train_iter, dev_iter, test_iter, args.data)
    else:
        test(config,model,test_iter, args.data)
        wandb.log({"preprocess_time":  float(pre_time_dif)})
        wandb.log({"averagepreprocess_time":  float(average_pre_time)})
        
    
if __name__ == '__main__':
    main()