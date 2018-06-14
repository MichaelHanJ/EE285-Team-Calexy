
# coding: utf-8

# In[1]:


from torchvision import models,transforms,datasets
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch.autograd import Variable
import torch
import time
from sklearn.svm import SVR
from collections import defaultdict


# # Data preprocessing

# In[2]:


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# In[3]:


prep1 = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])


# In[4]:


problem = 'age_balance'


# In[5]:


data_dir = os.path.join('Dataset', 'Labeled')
# data_dir = os.path.join('Dataset', 'LabelBy5')
# data_dir = os.path.join('Dataset', 'LabelRace')


# In[6]:


# os.removedirs(os.path.join(data_dir, '.ipynb_checkpoints'))


# In[7]:


dsets = datasets.ImageFolder(data_dir, prep1)


# In[8]:


dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=64, shuffle=False, num_workers=6)


# In[9]:


len(dset_loaders)


# In[10]:


len(dsets.classes)


# In[11]:


dsets.class_to_idx


# In[12]:


use_gpu = torch.cuda.is_available()


# # VGG

# In[13]:


model_vgg = models.vgg16(pretrained=True)
for param in model_vgg.parameters():
    param.requires_grad = False


# In[14]:


model_vgg


# In[15]:


model_vgg.classifier


# In[16]:


model_vgg = model_vgg.cuda()


# In[17]:


def preconvfeat(dataset):
    conv_features = []
    labels_list = []
    for data in dataset:
        inputs,labels = data

        inputs , labels = Variable(inputs.cuda()),Variable(labels.cuda())
        x = model_vgg.features(inputs)
        conv_features.extend(x.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
    conv_features = np.concatenate([[feat] for feat in conv_features])
    return (conv_features,labels_list)


# In[207]:


get_ipython().run_cell_magic(u'time', u'', u'conv_feat,labels = preconvfeat(dset_loaders)')


# In[209]:


# Saving the result
np.save('Features/conv_feat_age',conv_feat)
# np.save('conv_feat_' + problem,conv_feat)
# np.save('train_feat_' + problem,conv_feat[:19000])
# np.save('valid_feat_' + problem,conv_feat[19000:])


# In[210]:


np.save('Labels/label_age',labels)
# np.save('label_' + problem,labels)
# np.save('train_label_' + problem,labels[:19000])
# np.save('valid_label_' + problem,labels[19000:])


# # Load features

# In[172]:


# train_feat = np.load('train_feat_' + problem + '.npy') 
# valid_feat = np.load('valid_feat_' + problem + '.npy')
# train_feat = np.load('train_feat_new.npy')
# valid_feat = np.load('valid_feat_new.npy')


# In[173]:


# labels_train = np.load('train_label_' + problem + '.npy') 
# labels_valid = np.load('valid_label_' + problem + '.npy') 
# labels_train = np.load('train_label_new.npy')
# labels_valid = np.load('valid_label_new.npy')


# In[211]:


conv = np.load('Features/conv_feat_age.npy')
label = np.load('Labels/label_age.npy')
# conv = np.load('conv_feat_' + problem + '.npy')
# label = np.load('label_' + problem + '.npy')

# shuffle
index = np.random.permutation(len(conv))
conv = conv[index]
label = label[index]


# In[212]:


# Split
train_feat = conv[:19000]
valid_feat = conv[19000:]
labels_train = label[:19000]
labels_valid = label[19000:]


# In[213]:


diction = defaultdict(int)
for i in labels_train:
    diction[i] += 1
diction


# In[214]:


plt.plot(diction.keys(),diction.values())


# # Balance the age range

# In[215]:


# balanced distribution
def b_distribution(label,class_num_new):
    '''
    Input: Old label , updated number of class
    Output: New label (under balanced distribution)
    '''
    assert isinstance(class_num_new,int)
    assert class_num_new>0
#     assert int(len(label)/max_range) < class_num_new
    
    from collections import defaultdict
    diction = defaultdict(int)
    for i in label:
        diction[i] += 1
    
    batch_size = int(len(label)/class_num_new)
    trans = defaultdict(list)
    onebyone = defaultdict(int)
    flag = 0
    for i in range(class_num_new):
        counter = 0
        range_control = 0 
        while (flag<len(diction)):
            onebyone[flag] = i
            counter += diction[flag]
            trans[i].append(flag)
            flag += 1
            range_control += 1
            if counter>batch_size or range_control>5:
                break;
    
#     return trans, onebyone
    
    # update labels
    new_label = [onebyone[d] for d in label]
    return len(trans.keys()),new_label, trans


# In[216]:


tup = b_distribution(label,25)


# In[217]:


tup[0]


# In[218]:


label_balanced = tup[1]


# In[219]:


tup[2]


# In[220]:


diction_balanced = defaultdict(int)
for i in label_balanced:
    diction_balanced[i] += 1


# In[221]:


diction_balanced


# In[30]:


plt.plot(diction_balanced.keys(),diction_balanced.values())


# In[33]:


labels_train = label_balanced[:19000]
labels_valid = label_balanced[19000:]


# # Data generator

# In[13]:


def data_gen(conv_feat,labels,batch_size=64,shuffle=True):
    labels = np.array(labels)
    if shuffle:
        index = np.random.permutation(len(conv_feat))
        conv_feat = conv_feat[index]
        labels = labels[index]
    for idx in range(0,len(conv_feat),batch_size):
        yield(conv_feat[idx:idx+batch_size],labels[idx:idx+batch_size])


# In[14]:


def train_model(model,size,conv_feat=None,labels=None,epochs=1,optimizer=None,train=True,shuffle=True):
    for epoch in range(epochs):
        batches = data_gen(conv_feat=conv_feat,labels=labels,shuffle=shuffle)
        total = 0
        running_loss = 0.0
        running_corrects = 0
        running_corrects_oneclassoff = 0
        for inputs,classes in batches:
            inputs , classes = Variable(torch.from_numpy(inputs).cuda()),Variable(torch.from_numpy(classes).cuda())
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
#             outputs = outputs.float()
#             classes = classes.float()
#             print(outputs.size())
#             print(classes.size())
            loss = criterion(outputs,classes)           
            if train:
                if optimizer is None:
                    raise ValueError('Pass optimizer for train mode')
                optimizer = optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
#             print outputs.size()
            _,preds = torch.max(outputs.data,1)
            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == classes.data)
            running_corrects_oneclassoff += torch.sum(torch.abs(preds - classes.data) <= 1)
        epoch_loss = running_loss / float(size)
        epoch_acc = running_corrects / float(size)
        epoch_acc_oneclassoff = running_corrects_oneclassoff / float(size)
        print('Loss: {:.4f} Acc: {:.4f} One-class-off Acc: {:.4f}'.format(
                     epoch_loss, epoch_acc,epoch_acc_oneclassoff))
    
    return epoch_acc


# # Training All FC layers

# In[36]:


my_classifier = nn.Sequential(*list(model_vgg.classifier.children())[:-1])
# my_classifier.add_module('6',nn.Linear(4096, 10, bias=True))
# my_classifier.add_module('7',nn.ReLU(inplace = True))
# my_classifier.add_module('6',nn.Linear(4096, len(dsets.classes), bias=True))
my_classifier.add_module('6',nn.Linear(4096, tup[0], bias=True))
# my_classifier.add_module('7',nn.Softmax(dim=1))
my_classifier = my_classifier.cuda()
my_classifier


# In[37]:


# my_classifier = model_vgg.classifier
# my_classifier[6].out_features = len(dsets.classes)
# my_classifier


# In[38]:


for param in my_classifier.parameters():
    param.requires_grad = True


# In[39]:


criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
lr = 0.01
optimizer = torch.optim.SGD(my_classifier.parameters(),lr = lr)


# In[62]:


# %%time
# (train_model(model=my_classifier,size=19000,conv_feat=train_feat,labels=labels_train,
#             epochs=30,optimizer=optimizer,train=True,shuffle=True))


# In[63]:


# train_model(conv_feat=valid_feat,labels=labels_valid,model=my_classifier
#             ,size=4708,train=False,shuffle=False)


# # Training with validate every epoch

# In[40]:


# Relation between training accuracy and validation accuracy
acc_train = []
acc_valid = []
epoch = 40
for i in range(epoch):
    print 'training once'
    acc_train.append(train_model(model=my_classifier,size=19000,conv_feat=train_feat,labels=labels_train,
                    epochs=1,optimizer=optimizer,train=True,shuffle=True))
    
    print 'validate once'
    acc_valid.append(train_model(conv_feat=valid_feat,labels=labels_valid,model=my_classifier
                    ,size=4708,train=False,shuffle=False))


# In[44]:


plt.plot(range(1,epoch+1),acc_train,label = 'training')
plt.plot(range(1,epoch+1),acc_valid,label = 'validation')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy with ' + problem + ' classification')
plt.legend()
plt.show()


# In[45]:


# convert the classifier to cpu() format
my_classifier = my_classifier.cpu()


# In[46]:


# Save the classifier
torch.save(my_classifier, 'classifier_' + problem + '.pt')


# # Testing with new inputs

# In[187]:


test_case = [Image.open(os.path.join('yolo_output3',i)).convert('RGB') for i in os.listdir('yolo_output3') ]


# In[210]:


test_case[0] 


# In[212]:


test_case[0]= test_case[0].rotate(-30)
test_case[0] 


# In[216]:


test_case[2]


# In[218]:


test_case[2] = test_case[2].rotate(-30)
test_case[2] 


# In[219]:


prep2 = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])


# In[220]:


test_case_tensor = [prep2(d).unsqueeze_(0) for d in test_case]


# In[221]:


input_concat = reduce(lambda x,y:torch.cat((x,y),0),test_case_tensor)


# In[222]:


model_age = torch.load('classifier_age_balance.pt')
model_gender = torch.load('classifier_gender.pt')
model_race = torch.load('classifier_race.pt')


# In[223]:


model_age = model_age.cuda()
model_gender = model_gender.cuda()
model_race = model_race.cuda()


# In[224]:


# Hashtable for predict labels
gender_dict = {0:'Male',1:'Female'}
race_dict = {0:'White',1:'Black',2:'Asian',3:'Indian',4:'Others'}
age_dict =  {0: [0],
             1: [1, 2, 3],
             2: [4, 5, 6, 7, 8, 9],
             3: [10, 11, 12, 13, 14, 15],
             4: [16, 17, 18, 19, 20],
             5: [21, 22, 23],
             6: [24, 25],
             7: [26, 27],
             8: [28, 29],
             9: [30, 31],
             10: [32, 33, 34],
             11: [35, 36, 37],
             12: [38, 39, 40, 41],
             13: [42, 43, 44, 45, 46],
             14: [47, 48, 49, 50, 51],
             15: [52, 53, 54, 55],
             16: [56, 57, 58, 59, 60, 61],
             17: [62, 63, 64, 65, 66, 67],
             18: [68, 69, 70, 71, 72, 73],
             19: [74, 75, 76, 77, 78, 79],
             20: [80, 81, 82, 83, 84, 85],
             21: [86, 87, 88, 89, 90, 91],
             22: [92, 93, 94, 95, 96, 97],
             23: [98, 99, 100, 101, 102, 103]}


# In[225]:


age_hash = {v:u for u,v in dsets.class_to_idx.items()}


# In[226]:


age_dict_new = defaultdict(list)
for u,v in age_dict.items():
    for i in v: 
        age_dict_new[u].append(age_hash[i])


# In[227]:


age_dict_update = {u:str(int(v[0]))+'~'+str(int(v[-1])) for u,v in age_dict_new.items()}
age_dict_update


# In[235]:


def test_perf(model,input_concat):
    inputs = Variable(input_concat.cuda())
    feat = model_vgg.features(inputs)
    inputs_2 = feat.view(feat.size(0), -1)
    outputs = model(inputs_2)
    _,pred = torch.max(outputs.data,1)
   
    return pred.cpu().numpy()


# In[236]:


predict_age = test_perf(model_age, input_concat)
predict_gender = test_perf(model_gender, input_concat)
predict_race = test_perf(model_race, input_concat)


# In[237]:


predict_age


# In[238]:


predict_gender


# In[239]:


predict_race


# In[240]:


result_gender = [gender_dict[d] for d in predict_gender]
result_race = [race_dict[d] for d in predict_race]
result_age = [age_dict_update[d] for d in predict_age]


# In[241]:


for i in range(len(test_case)):
    f,ax = plt.subplots()
    ax.imshow(test_case[i])
    ax.set_title(result_gender[i] + ' ' + result_race[i] + ' ' + str(result_age[i]))


# In[242]:


result_age


# In[113]:


loading = torch.load('classifier_balanced.pt')


# In[114]:


loading


# In[115]:


loading = nn.Sequential(*list(loading.children())[:-1])
# loading.add_module('6',nn.Linear(4096, 4, bias=True))
# loading.add_module('7',nn.Softmax(dim=1))
loading.add_module('6',nn.Linear(4096, 1, bias=True))
loading.add_module('7',nn.ReLU())
loading = loading.cuda()
loading


# In[116]:


conv = np.load('conv_feat.npy')
label = np.load('label.npy')
index = np.random.permutation(len(conv))
conv = conv[index]
label = label[index]
# Split
train_feat = conv[:19000]
valid_feat = conv[19000:]
labels_train = label[:19000]
labels_valid = label[19000:]


# In[ ]:


# initialize 


# In[117]:


batches = data_gen(conv_feat=train_feat,labels=labels_train,shuffle=True)
total = 0
feat_np = []
label_np = []
check = {y:x for x,y in dsets.class_to_idx.iteritems()}

for inputs,classes in batches:
    inputs = Variable(torch.from_numpy(inputs).cuda())
    inputs = inputs.view(inputs.size(0), -1)
    outputs = loading(inputs)
    feat_np.append(outputs.data.cpu().numpy())
    temp = [int(check[d]) for d in classes]
    label_np.append(temp)
    


# In[118]:


feat_np = np.array(feat_np)
label_np = np.array(label_np)


# In[119]:


aa = reduce(lambda x,y:x+y, label_np)
aa = np.array(aa)


# In[120]:


aa.shape


# In[121]:


bb = reduce(lambda x,y:np.concatenate((x,y),axis=0),feat_np)


# In[122]:


bb.shape


# In[123]:


def mse(data, pred):
    return np.sum((data - pred)**2)/len(data)


# In[124]:


svr_lin = SVR(kernel='linear', C=1e3)


# In[125]:


svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)


# In[126]:


predict_train = svr_rbf.fit(bb, aa).predict(bb)


# In[128]:


mse_rbf = mse(aa,predict_train)
print mse_rbf


# In[132]:


predict_train_lin = svr_lin.fit(bb, aa).predict(bb)


# In[133]:


mse_lin = mse(aa,predict_train_lin)
print mse_lin


# In[134]:


aa


# In[135]:


predict_train


# In[136]:


predict_train_lin


# In[139]:


X = bb
lw = 2
plt.scatter(X, aa, color='darkorange', label='data')
plt.scatter(X, predict_train, color='navy', label='RBF model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


# In[138]:


plt.scatter(X[:1000], aa[:1000], color='darkorange', label='data')


# In[104]:


aa.min()


# In[168]:


aa

