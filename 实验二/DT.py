from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz
from collections import Counter
import numpy as np
import os, re

# graphviz的安装路径
os.environ["PATH"] += os.pathsep + r'D:\Graphviz\bin'
train = 'traindata.txt'
test = 'testdata.txt'
train_X = []
train_y = []
test_X = []
test_y = []
labels = ['first', 'second', 'third', 'fourth']
with open(train, 'r') as f:
    for line in f.readlines():
        data = re.split('\t', line.strip())
        if (data[0] != 'traindata=[' and data[0] != '];'):
            train_X.append([float(i) for i in data[:-1]])
            train_y.append(int(data[-1]))

with open(test, 'r') as f:
    for line in f.readlines():
        data = re.split('\t', line.strip())
        if (data[0] != 'testdata=[' and data[0] != '];'):
            test_X.append([float(i) for i in data[:-1]])
            test_y.append(int(data[-1]))

X = MinMaxScaler().fit_transform(np.array(train_X))

print('数据分布：', Counter(train_y))

## 决策树
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(train_X, train_y)
predict_y = clf.predict(test_X)

print('测试集平均准确率：', clf.score(test_X, test_y))
print('测试结果:', predict_y)

dot_data = export_graphviz(clf, feature_names=labels, class_names=['1', '2', '3'], filled=True, out_file='DT.dot',
                           rounded=True)

os.system('dot -Tpng DT.dot -o DT.png')  # dot -Tpng RF.dot -o RF.png
# 中文
# f_old = open('DT_old.dot', 'r', encoding='utf-8')
# f_new = open('DT.dot', 'w', encoding='utf-8')
# for line in f_old:
#     if 'fontname' in line:
#         font_re = 'fontname=(.*?)]'
#         old_font = re.findall(font_re, line)[0]
#         line = line.replace(old_font, 'SimHei')
#     f_new.write(line)
# f_old.close()
# f_new.close()
