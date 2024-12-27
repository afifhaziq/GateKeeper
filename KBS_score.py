from importlib import import_module
import torch
import torch.nn.functional as F


# Your dataset path
dataset =  "/media/jie/MyPassport/ExpBackup/11.26-3080/Program/GateKeeper/FNet/dataset/IoT23"
model_name = "Base"
x = import_module(model_name)
config = x.Config(dataset)
model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path),strict=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
class_nums =len([x.strip() for x in open(dataset + '/data/class.txt').readlines()])

def toTensor(x):
    # Split and convert to integer list
    x = [int(i.strip("\n")) for i in x.split()][:config.max_byte_len]
    
    # Convert to tensor and move to target device
    x = torch.LongTensor(x).to(device)
    
    # Generate position encoding
    pos = torch.arange(config.max_byte_len, dtype=torch.long).to(device)
    # Adjust dimensions
    x = x.view(1, config.max_byte_len)
    pos = pos.view(1, config.max_byte_len)
    
    return x, pos




def eval(traffic,pos):
    model.eval()
    with torch.no_grad():
        outputs,score = model(traffic,pos)
        preds = F.softmax(outputs,dim=1).squeeze()
        pred_label = torch.argmax(outputs.data, 1).cpu().numpy()
        return score.squeeze()




def main():
    # Initialize a dictionary to store rankings for each category
    f_score = open("attention_score/" + dataset.split("/")[-2] + "KBS_result.csv", "w")
    dict_rank = {str(i): torch.zeros(config.max_byte_len).to(device) 
                 for i in range(class_nums)}
    
    # Read validation dataset
    with open(dataset + "/data/dev.txt", 'r') as fdev:
        samples_dev = fdev.readlines()

    # Calculate attention scores for all samples
    count = torch.zeros(config.max_byte_len).to(device)
    for line in samples_dev:
        traffic_str, label = line.strip().split("\t")
        traffic, pos = toTensor(traffic_str)
        score = eval(traffic, pos)
        dict_rank[label.strip()] += score
        count += score
    
    # Normalize the scores and convert them to a list
    count = (count / torch.max(count)).tolist()
    
    # Generate ranking information
    rank = [(pos, score) for pos, score in enumerate(count)]
    
    # Write raw scores
    ranked_scores = [str(score) for _, score in rank]
    f_score.write(",".join(ranked_scores) + "\n")
    
    # Sort by scores and write rankings
    rank_sorted = sorted(rank, key=lambda x: x[1], reverse=True)
    ranked_scores = [score for _, score in rank_sorted]
    rank_positions = [str(ranked_scores.index(score) + 1) for _, score in rank]
    f_score.write(",".join(rank_positions) + "\n")
    
    # Get positions corresponding to the highest scores
    top_positions = [pos for pos, _ in rank_sorted]
    print(top_positions)
   
   
if __name__ == "__main__":
    main()

    
