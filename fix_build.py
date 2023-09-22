import json 

with open('datasets/build/dev.json') as f:
    data_dict = json.load(f)
    
for i in data_dict:
    doc = data_dict[i]
    data_dict[i]['label_ids']= [i[0] for i in data_dict[i]['label_ids']]

with open('datasets/build/dev.json', "w") as f:
    json.dump(data_dict, f)


with open('datasets/build/train.json') as f:
    data_dict = json.load(f)
for i in data_dict:
    doc = data_dict[i]
    data_dict[i]['label_ids']= [i[0] for i in data_dict[i]['label_ids']]

with open('datasets/build/train.json', "w") as f:
     json.dump(data_dict, f)