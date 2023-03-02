# import pandas as pd
# import torch
# import numpy as np
# import random
#
# class LogisticNet(torch.nn.Module):
#     def __init__(self,n_features):
#         super(LogisticNet,self).__init__()
#         self.linear=torch.nn.Linear(n_features,1)
#         self.sigmoid=torch.nn.Sigmoid()
#     # forward 定义前向传播
#     def forward(self, x):
#         x = self.linear(x)
#         output = self.sigmoid(x)
#         return output
#
# if __name__ == '__main__':
#     # x,y是矩阵，3行1列 也就是说总共有3个样本，每个样本只有1个特征
#     train_data = pd.read_csv('./output3/behavior_data.csv')
#     train_data = np.array(train_data)
#     train_x = train_data[:,1:3]
#     train_y = train_data[:,3]
#     print(train_x)
#     print(train_y)
#     x = torch.tensor(train_x).to(torch.float32)
#     y = torch.tensor(train_y).unsqueeze(1)
#     y = y.to(torch.float32)
#
#     # Step1: 创建模型
#     model = LogisticNet(n_features=2)
#
#     # Step2: 创建损失函数和优化器
#     Loss = torch.nn.BCELoss()
#     optimizer = torch.optim.SGD(params=model.parameters(), lr=0.05)
#
#     # Step3: 训练
#     print(x)
#     print(y)
#     for epoch in range(70):
#         y_pred = model.forward(x)
#         loss = Loss(y_pred, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     x_test = torch.Tensor([[8.0,98.0]])
#     y_test = model(x_test)
#     print('y_pred = ', y_test.data)
#
#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def train_model():
    train_data = pd.read_csv('./output3/behavior_data.csv')
    train_data = np.array(train_data)



    sc = StandardScaler()
    sc.fit(train_data[:,1:3])
    X_train_std = sc.transform(train_data[:,1:3])

    LR = LogisticRegression(C=1.0, penalty='l2', tol=0.01, solver='liblinear',dual=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train_std, train_data[:,3], test_size=0.2, random_state=42)
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    LR.fit(X_res,y_res)


    # clf = MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(2, 1), random_state=1)
    # clf.fit(X_train_std, y_train)
    # print(LR.predict_proba([[float("inf"),float("inf")]]))
    # print("train_set")
    y_true = y_train
    y_pred = LR.predict(X_train)
    target_names = ['0', '1']
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("val_set")
    y_true = y_test
    y_pred = LR.predict(X_test)
    print(classification_report(y_true, y_pred, target_names=target_names))
    return LR
if __name__ == '__main__':
    lr_model = train_model()
    distance_list = list(range(100,300,1))
    price_list = list(range(0,1000))
    prob_array = np.empty(shape=(0,0))
    num = 0
    temp = np.array([[1,2],[12,2],[1.1,2.9],[1,2]])
    print(temp.shape)
    price_array = np.random.random((300,300))
    dis_array = 2*price_array
    # print("the predict is",lr_model.predict_proba(temp))
    temp_ = np.dstack((price_array,dis_array))
    temp_ = temp_.reshape(-1,2)
    print(temp_)
    res = lr_model.predict_proba(temp_)
    res = res.reshape(300,300,2)
    print(res.shape)
    res_ = np.delete(res,0,axis=2)
    print("the predict is", res)
    print("the predict is", res_)
    for dis_val in distance_list:
        for price_val in price_list:
            temp_list = []
            for i in range(10):
                temp_list.append(lr_model.predict_proba([[dis_val*0.1,price_val*0.1]])[0,1])
            # print(np.mean(temp_list))
            prob_array = np.append(prob_array,np.mean(temp_list))
    prob_array = prob_array.reshape(len(distance_list),len(price_list))
    print(prob_array.shape)
    b = ['%.2f' % (i * 0.1) for i in distance_list]
    p = ['%.2f' % (i * 0.1) for i in price_list]
    prob_pd = pd.DataFrame(prob_array,index=b,columns=p)
    print(prob_pd)


    ax = sns.heatmap(prob_pd,cmap="YlGnBu")
    # ax = sns.heatmap(prob_array,ax=distance_list)
    plt.xticks(fontsize=10)  # x轴刻度的字体大小（文本包含在pd_data中了）
    plt.yticks(fontsize=10)  # y轴刻度的字体大小（文本包含在pd_data中了）
    plt.xlabel('Price', fontsize=10, color='k')  # x轴label的文本和字体大小
    plt.ylabel('Distance', fontsize=10, color='k')  # y轴label的文本和字体大小
    plt.show()


