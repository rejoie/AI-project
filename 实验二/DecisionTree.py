import copy
from math import log, inf
import operator, re, os
import numpy as np
import pandas as pd
from matplotlib import colors
from collections import Counter
from treePlotter import createPlot

# graphviz的安装路径
os.environ["PATH"] += os.pathsep + r'D:\Graphviz\bin'
color = ['lightcoral', 'coral', 'darkorange', 'gold', 'palegreen', 'paleturquoise', 'skyblue', 'plum', 'hotpink',
         'pink']
hexColor = [colors.cnames[i] for i in color]

print(hexColor)
train = 'traindata.txt'
test = 'testdata.txt'
train_data = []
test_data = []
labels = ['first', 'second', 'third', 'fourth']
Class = [1, 2, 3]
labelProperties = [1, 1, 1, 1]
dotName = []

with open(train, 'r') as f:
    for line in f.readlines():
        data = re.split('\t', line.strip())
        if (data[0] != 'traindata=[' and data[0] != '];'):
            train_data.append([float(i) for i in data])

with open(test, 'r') as f:
    for line in f.readlines():
        data = re.split('\t', line.strip())
        if (data[0] != 'testdata=[' and data[0] != '];'):
            test_data.append([float(i) for i in data])

train_data = np.array(train_data)
test_data = np.array(test_data)


def calcShannonEnt(dataset):  # 计算数据的熵(entropy)
    numEntries = len(dataset)  # 数据条数
    labelCounts = Counter([i[-1] for i in dataset])  # 统计有多少个类以及每个类的数量
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 计算单个类的熵值
        shannonEnt -= prob * log(prob, 2)  # 累加每个类的熵值
    return shannonEnt


# 划分数据集，axis:按第几个属性划分，value:要返回的子集对应的属性值
def splitDataset(dataset, axis, value):  # 按某个特征分类后的数据
    retDataset = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataset.append(reducedFeatVec)
    return retDataset


# 划分数据集, axis:按第几个特征划分, value:划分特征的值, LorR: value值左侧（小于）或右侧（大于）的数据集
def splitDataset_c(dataset, axis, value, LorR='L'):
    retdataset = []
    if LorR == 'L':
        for featVec in dataset:
            if float(featVec[axis]) < value:
                retdataset.append(featVec)
    else:
        for featVec in dataset:
            if float(featVec[axis]) > value:
                retdataset.append(featVec)
    return np.array(retdataset)


# # 选择最好的数据集划分方式C4.5
# def chooseBestFeatureToSplit(dataset, labelProperty):
#     baseEntropy = calcShannonEnt(dataset)  # 计算根节点的信息熵
#     bestInfoGain = 0
#     newEntropyRatio = 0
#     bestInfoGainRatio = 0.0
#     bestFeature = -1
#     bestPartValue = None  # 连续的特征值，最佳划分值
#     count = {}
#     for i in range(len(labelProperty)):  # 对每个特征循环
#         featList = dataset[:, i]  # [example[i] for example in dataset]
#         uniqueVals = set(featList)  # 该特征包含的所有值
#         newEntropy = 0.0
#         H = 0.0
#         bestPartValuei = None
#         if labelProperty[i] == 0:  # 对离散的特征
#             for value in uniqueVals:  # 对每个特征值，划分数据集, 计算各子集的信息熵
#                 subDataset = splitDataset(dataset, i, value)
#                 prob = len(subDataset) / float(len(dataset))
#                 newEntropy += prob * calcShannonEnt(subDataset)
#                 H += -prob * log(prob, 2)
#             newEntropyRatio = newEntropy / (H + 1e-8)
#
#         else:  # 对连续的特征
#             sortedUniqueVals = list(uniqueVals)  # 对特征值排序
#             sortedUniqueVals.sort()
#             maxRatio = -inf
#             for j in range(len(sortedUniqueVals) - 1):  # 计算划分点
#                 partValue = (float(sortedUniqueVals[j]) + float(
#                     sortedUniqueVals[j + 1])) / 2
#                 # 对每个划分点，计算信息熵
#                 datasetLeft = splitDataset_c(dataset, i, partValue, 'L')
#                 datasetRight = splitDataset_c(dataset, i, partValue, 'R')
#                 probLeft = len(datasetLeft) / float(len(dataset))
#                 probRight = len(datasetRight) / float(len(dataset))
#                 Entropy = probLeft * calcShannonEnt(datasetLeft) + probRight * calcShannonEnt(datasetRight)
#                 H = -probLeft * log(probLeft, 2) - probRight * log(probRight, 2)
#                 infoGainRatio = (baseEntropy - Entropy) / (H + 1e-8)
#                 if infoGainRatio > maxRatio:  # 取最大的信息增益率
#                     maxRatio = infoGainRatio
#                     bestPartValuei = partValue
#             newEntropyRatio = maxRatio
#         if newEntropyRatio > bestInfoGain:  # 取最大的信息增益对应的特征
#             bestInfoGainRatio = newEntropyRatio
#             bestFeature = i
#             bestPartValue = bestPartValuei
#
#     if labelProperty[bestFeature] == 0:
#         for value in set(dataset[:, bestFeature]):
#             datasetPart = []
#             for item in dataset:
#                 if item[bestFeature] == value:
#                     datasetPart.append(item[-1])
#             count[value] = list(Counter(datasetPart).values())
#     else:
#         if (bestPartValue):
#             datasetLeft = splitDataset_c(dataset, bestFeature, bestPartValue, 'L')
#             datasetRight = splitDataset_c(dataset, bestFeature, bestPartValue, 'R')
#             count = {'L': list(Counter([i[-1] for i in datasetLeft]).values()),
#                      'R': list(Counter([i[-1] for i in datasetRight]).values())}
#         else:
#             count = {'L': [0] * 3, 'R': [0] * 3}
#
#     return bestFeature, bestPartValue, bestInfoGainRatio, count


# 选择最好的数据集划分方式ID3
def chooseBestFeatureToSplit(dataset, labelProperty):
    baseEntropy = calcShannonEnt(dataset)  # 计算根节点的信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    bestPartValue = None  # 连续的特征值，最佳划分值
    count = {}
    for i in range(len(labelProperty)):  # 对每个特征循环
        featList = dataset[:, i]  # [example[i] for example in dataset]
        uniqueVals = set(featList)  # 该特征包含的所有值
        newEntropy = 0.0
        bestPartValuei = None
        if labelProperty[i] == 0:  # 对离散的特征
            for value in uniqueVals:  # 对每个特征值，划分数据集, 计算各子集的信息熵
                subDataset = splitDataset(dataset, i, value)
                prob = len(subDataset) / float(len(dataset))
                newEntropy += prob * calcShannonEnt(subDataset)
        else:  # 对连续的特征
            sortedUniqueVals = list(uniqueVals)  # 对特征值排序
            sortedUniqueVals.sort()
            minEntropy = inf
            for j in range(len(sortedUniqueVals) - 1):  # 计算划分点
                partValue = (float(sortedUniqueVals[j]) + float(
                    sortedUniqueVals[j + 1])) / 2
                # 对每个划分点，计算信息熵
                datasetLeft = splitDataset_c(dataset, i, partValue, 'L')
                datasetRight = splitDataset_c(dataset, i, partValue, 'R')
                probLeft = len(datasetLeft) / float(len(dataset))
                probRight = len(datasetRight) / float(len(dataset))
                Entropy = probLeft * calcShannonEnt(datasetLeft) + probRight * calcShannonEnt(datasetRight)
                if Entropy < minEntropy:  # 取最小的信息熵
                    minEntropy = Entropy
                    bestPartValuei = partValue
            newEntropy = minEntropy
        infoGain = baseEntropy - newEntropy  # 计算信息增益
        if infoGain > bestInfoGain:  # 取最大的信息增益对应的特征
            bestInfoGain = infoGain
            bestFeature = i
            bestPartValue = bestPartValuei

    if labelProperty[bestFeature] == 0:
        for value in set(dataset[:, bestFeature]):
            datasetPart = []
            for item in dataset:
                if item[bestFeature] == value:
                    datasetPart.append(item[-1])
            count[value] = list(Counter(datasetPart).values())
    else:
        if (bestPartValue):
            datasetLeft = splitDataset_c(dataset, bestFeature, bestPartValue, 'L')
            datasetRight = splitDataset_c(dataset, bestFeature, bestPartValue, 'R')
            count = {'L': list(Counter([i[-1] for i in datasetLeft]).values()),
                     'R': list(Counter([i[-1] for i in datasetRight]).values())}
        else:
            count = {'L': [0] * 3, 'R': [0] * 3}

    return bestFeature, bestPartValue, bestInfoGain, count


def majorityCnt(classList):  # 按分类后类别数量排序
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def countClass(data, Class):
    dic = Counter(data)
    for i in Class:
        if i not in dic.keys():
            dic[i] = 0

    return tuple([dic[i] for i in Class])


# 创建树, 样本集 特征 特征属性（0 离散， 1 连续）
def createTree(dataset, labels, labelProperty, Class, limitDepth, depth, name, fatherName, method):
    classList = dataset[:, -1]  # 类别向量
    if (limitDepth <= depth):
        C = majorityCnt(classList)
        fatherName = name
        name = str(C)
        with open('Tree.dot', 'a+') as f:
            f.write('{} [label="{} = {}\\nsamples = {}\\nvalue = {}\\nclass = {}", fillcolor="{}"] ;\n'.format(
                fatherName + name, method, round(calcShannonEnt(dataset), 4), len(dataset),
                countClass(classList, Class), int(C), hexColor[np.random.randint(0, len(hexColor) - 1)]))
        dotName.append(fatherName + name)
        return int(C), name
    if set(classList) == len(classList):  # 如果只有一个类别，返回
        fatherName = name
        name = str(classList[0])
        with open('Tree.dot', 'a+') as f:
            f.write('{} [label="{} = {}\\nsamples = {}\\nvalue = {}\\nclass = {}", fillcolor="{}"] ;\n'.format(
                fatherName + name, method, round(calcShannonEnt(dataset), 4), len(dataset),
                countClass(classList, Class), int(classList[0]), hexColor[np.random.randint(0, len(hexColor) - 1)]))
        dotName.append(fatherName + name)
        return int(classList[0]), name
    if len(dataset[0]) == 1:  # 如果所有特征都被遍历完了，返回出现次数最多的类别
        C = majorityCnt(classList)
        fatherName = name
        name = str(C)
        with open('Tree.dot', 'a+') as f:
            f.write('{} [label="{} = {}\\nsamples = {}\\nvalue = {}\\nclass = {}", fillcolor="{}"] ;\n'.format(
                fatherName + name, method, round(calcShannonEnt(dataset), 4), len(dataset),
                countClass(classList, Class), int(C), hexColor[np.random.randint(0, len(hexColor) - 1)]))
        dotName.append(fatherName + name)
        return int(C), name
    # 获得最优特征、最优分段点、最优ID3，左边统计，右边统计
    bestFeat, bestPartValue, bestInfoGain, count = chooseBestFeatureToSplit(dataset, labelProperty)  # 最优分类特征的索引
    if bestFeat == -1:  # 如果无法选出最优分类特征，返回出现次数最多的类别
        C = majorityCnt(classList)
        fatherName = name
        name = str(C)
        with open('Tree.dot', 'a+') as f:
            f.write('{} [label="{} = {}\\nsamples = {}\\nvalue = {}\\nclass = {}", fillcolor="{}"] ;\n'.format(
                fatherName + name, method, round(calcShannonEnt(dataset), 4), len(dataset),
                countClass(classList, Class), int(C), hexColor[np.random.randint(0, len(hexColor) - 1)]))
        dotName.append(fatherName + name)
        return int(C), name
    if labelProperty[bestFeat] == 0:  # 对离散的特征
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        labelsNew = copy.copy(labels)
        labelPropertyNew = copy.copy(labelProperty)
        del (labelsNew[bestFeat])  # 已经选择的特征不再参与分类
        del (labelPropertyNew[bestFeat])
        featValues = dataset[:, bestFeat]
        uniqueValue = set(featValues)  # 该特征包含的所有值
        fatherName = name
        name = bestFeatLabel
        with open('Tree.dot', 'a+') as f:
            f.write(
                '{} [label="{}\\n{} = {}\\nsamples = {}\\nvalue = {}\\n", fillcolor="{}"] ;\n'.format(fatherName + name,
                                                                                                      bestFeatLabel,
                                                                                                      method, round(
                        calcShannonEnt(dataset), 4), len(dataset), countClass(classList, Class), hexColor[
                                                                                                          np.random.randint(
                                                                                                              0, len(
                                                                                                                  hexColor) - 1)]))
        dotName.append(fatherName + name)
        for value in uniqueValue:  # 对每个特征值，递归构建树
            subLabels = labelsNew[:]
            subLabelProperty = labelPropertyNew[:]
            myTree[bestFeatLabel][value], nextNodeName = createTree(splitDataset(dataset, bestFeat, value), subLabels,
                                                                    subLabelProperty, Class, limitDepth, depth + 1,
                                                                    name, fatherName, method)
            with open('Tree.dot', 'a+') as f:
                f.write('{} -> {} ;\n'.format(fatherName + name, name + nextNodeName))
    else:  # 对连续的特征，不删除该特征，分别构建左子树和右子树
        bestFeatLabel = labels[bestFeat] + ' < ' + str(round(bestPartValue, 4))
        myTree = {bestFeatLabel: {}}
        subLabels = labels[:]
        subLabelProperty = labelProperty[:]
        fatherName = name
        name = labels[bestFeat] + str(round(bestPartValue, 4))
        with open('Tree.dot', 'a+') as f:
            f.write(
                '{} [label="{}\\n{} = {}\\nsamples = {}\\nvalue = {}\\n", fillcolor="{}"] ;\n'.format(fatherName + name,
                                                                                                      bestFeatLabel,
                                                                                                      method, round(
                        calcShannonEnt(dataset), 4), len(dataset), countClass(classList, Class), hexColor[
                                                                                                          np.random.randint(
                                                                                                              0, len(
                                                                                                                  hexColor) - 1)]))
        dotName.append(fatherName + name)
        # 构建左子树
        valueLeft = 'yes'
        myTree[bestFeatLabel][valueLeft], nextNodeName = createTree(
            splitDataset_c(dataset, bestFeat, bestPartValue, 'L'), subLabels, subLabelProperty, Class, limitDepth,
            depth + 1, name, fatherName, method)
        with open('Tree.dot', 'a+') as f:
            f.write('{} -> {} [labeldistance=2.5, labelangle=45, headlabel="Yes"] ;\n'.format(fatherName + name,
                                                                                              name + nextNodeName))

        # 构建右子树
        valueRight = 'no'
        myTree[bestFeatLabel][valueRight], nextNodeName = createTree(
            splitDataset_c(dataset, bestFeat, bestPartValue, 'R'), subLabels, subLabelProperty, Class, limitDepth,
            depth + 1, name, fatherName, method)
        with open('Tree.dot', 'a+') as f:
            f.write('{} -> {} [labeldistance=2.5, labelangle=45, headlabel="No"] ;\n'.format(fatherName + name,
                                                                                             name + nextNodeName))

    return myTree, name


# 更改dot内变量名
def changeDot(path, dotName):
    f_old = open(path, 'r', encoding='utf-8')
    f_new = open('new' + path, 'w', encoding='utf-8')
    for line in f_old:
        for i in range(len(dotName)):
            if dotName[i] in line:
                line = line.replace(dotName[i], str(i))
        f_new.write(line)
    f_old.close()
    f_new.close()


# 测试算法
def classify(inputTree, classList, featLabels, featLabelProperties, testVec):
    firstStr = list(inputTree.keys())[0]  # 根节点
    firstLabel = firstStr
    lessIndex = str(firstStr).find(' < ')
    if lessIndex > -1:  # 如果是连续型的特征
        firstLabel = str(firstStr)[:lessIndex]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstLabel)  # 跟节点对应的特征
    for key in secondDict.keys():  # 对每个分支循环
        if featLabelProperties[featIndex] == 0:  # 离散的特征
            if testVec[featIndex] == key:  # 测试样本进入某个分支
                if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    return classify(secondDict[key], classList, featLabels, featLabelProperties, testVec)
                else:  # 如果是叶子， 返回结果
                    return secondDict[key]
            elif testVec[featIndex] == 'Nan':  # 如果测试样本的属性值缺失，则进入每个分支
                if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    return classify(secondDict[key], classList, featLabels, featLabelProperties, testVec)
                else:  # 如果是叶子， 返回结果
                    return secondDict[key]
        else:
            partValue = float(str(firstStr)[lessIndex + 3:])
            if testVec[featIndex] == 'Nan':  # 如果测试样本的属性值缺失，则对每个分支的结果加和
                # 进入左子树
                if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    return classify(secondDict[key], classList, featLabels, featLabelProperties, testVec)
                else:  # 如果是叶子， 返回结果
                    return secondDict[key]
            elif float(testVec[featIndex]) <= partValue and key == 'yes':  # 进入左子树
                if type(secondDict['yes']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    return classify(secondDict['yes'], classList, featLabels, featLabelProperties, testVec)
                else:  # 如果是叶子， 返回结果
                    return secondDict[key]
            elif float(testVec[featIndex]) > partValue and key == 'no':
                if type(secondDict['no']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                    return classify(secondDict['no'], classList, featLabels, featLabelProperties, testVec)
                else:  # 如果是叶子， 返回结果
                    return secondDict[key]


# 测试决策树正确率
def testing(myTree, classList, data_test, labels, labelProperties):
    error = 0
    for i in range(len(data_test)):
        classLabel = classify(myTree, classList, labels, labelProperties, data_test[i, :-1])
        if classLabel != data_test[i][-1]:
            error += 1
    return error


if __name__ == '__main__':
    with open('Tree.dot', 'a+') as f:
        f.write(
            'digraph Tree {\nnode [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;\nedge [fontname=helvetica] ;\n')
    myTree, _ = createTree(train_data, labels, labelProperties, Class, 3, 1, str(0), str(-1), 'entropy')
    with open('Tree.dot', 'a+') as f:
        f.write('}\n')
    print(myTree)  # 输出决策树模型结果
    changeDot('Tree.dot', list(set(dotName)))
    createPlot(myTree)
    os.system('dot -Tpng newTree.dot -o Tree.png')

    pred = []
    for item in test_data:
        pred.append(classify(myTree, Class, labels, labelProperties, item[:-1]))

    print(pred)
    print('错误率：', testing(myTree, Class, test_data, labels, labelProperties) / len(test_data))
