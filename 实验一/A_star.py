import numpy as np
from copy import deepcopy
from math import inf, sqrt

rightPos = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]
# rightPos = [[1, 2, 3, 4, 5], [16, 17, 18, 19, 6], [15, 24, 0, 20, 7], [14, 23, 22, 21, 8], [13, 12, 11, 10, 9]]
Open = []
Closed = []


# 判断两状态是否相等
def judgeSame(state1, state2):
    # for i in range(3):
    #     for j in range(3):
    #         if state1[i][j] != state2[i][j]:
    #             return False
    # return True
    return str(state1) == str(state2)


# 打印输出
def printState(state, dim):
    state_str = [' ' * len(str(dim * dim)) if i == 0 else str(i) + ' ' * (len(str(dim * dim)) - len(str(i))) for i in
                 np.array(state).flatten()]
    for i in range(dim):
        print(' '.join(state_str[dim * i:dim * i + dim]))


# 寻找数字的索引位置
def findIndex(state, num, dim):
    for i in range(dim):
        for j in range(dim):
            if (state[i][j] == num):
                return (i, j)
    return -1


# 计算f函数，不计算0
def computeF(state, step, dim):
    h = 0
    for i in range(dim):
        for j in range(dim):
            if (state[i][j] != 0):
                row, col = findIndex(rightPos, state[i][j], dim)
                h += abs(i - row) + abs(j - col)

    return h + step


# # 放错数字个数
# def computeF(state, step, dim):
#     h = 0
#     for i in range(dim):
#         for j in range(dim):
#             if ((i, j) != findIndex(rightPos, state[i][j], dim) and state[i][j] != 0):
#                 h += 1
#     return h + step


# # 欧几里得距离
# def computeF(state, step, dim):
#     h = 0
#     for i in range(dim):
#         for j in range(dim):
#             if (state[i][j] != 0):
#                 row, col = findIndex(rightPos, state[i][j], dim)
#                 h += sqrt((i - row) ** 2 + (j - col) ** 2)
#
#     return h + step


def getInvCount(state, dim):
    temp = np.array(state).flatten()

    inverseNum = 0
    for i in range(dim * dim):
        for j in range(i + 1, dim * dim):
            if (temp[j] != 0 and temp[i] != 0 and temp[i] > temp[j]):
                inverseNum += 1

    return inverseNum


def isSolvable(initState, rightPos, dim):
    return getInvCount(initState, dim) % 2 == getInvCount(rightPos, dim) % 2


def haveArrived(state):
    for i in Closed:
        if (judgeSame(i[0], state)):
            return True
    return False


# 计算每个方向的估价值，并添加至open表
def moveAndAdd(state, step, dim):
    # 找出0的最小值
    i, j = findIndex(state, 0, dim)
    value = []
    newState = []
    # direction=='up'
    if (i == 0):
        value.append(inf)
        newState.append([])
    else:
        state_up = deepcopy(state)
        state_up[i][j], state_up[i - 1][j] = state_up[i - 1][j], state_up[i][j]
        if (not haveArrived(state_up)):
            value.append(computeF(state_up, step, dim))
            newState.append(state_up)
        else:
            value.append(inf)
            newState.append([])
    # direction=='down'
    if (i == dim - 1):
        value.append(inf)
        newState.append([])
    else:
        state_down = deepcopy(state)
        state_down[i][j], state_down[i + 1][j] = state_down[i + 1][j], state_down[i][j]
        if (not haveArrived(state_down)):
            value.append(computeF(state_down, step, dim))
            newState.append(state_down)
        else:
            value.append(inf)
            newState.append([])
    # direction=='left'
    if (j == 0):
        value.append(inf)
        newState.append([])
    else:
        state_left = deepcopy(state)
        state_left[i][j], state_left[i][j - 1] = state_left[i][j - 1], state_left[i][j]
        if (not haveArrived(state_left)):
            value.append(computeF(state_left, step, dim))
            newState.append(state_left)
        else:
            value.append(inf)
            newState.append([])
    # direction=='right'
    if (j == dim - 1):
        value.append(inf)
        newState.append([])
    else:
        state_right = deepcopy(state)
        state_right[i][j], state_right[i][j + 1] = state_right[i][j + 1], state_right[i][j]
        if (not haveArrived(state_right)):
            value.append(computeF(state_right, step, dim))
            newState.append(state_right)
        else:
            value.append(inf)
            newState.append([])

    # 将新状态加入表，添加到open表
    for i in range(4):
        if (value[i] < inf):
            Open.append((newState[i], state, value[i], step + 1))


def A_Star(state, dim):
    Open.append((state, -1, computeF(state, 0, dim), 0))
    while (not judgeSame(state, rightPos)):
        Min = sorted(Open, key=lambda x: x[2])[0]
        Open.remove(Min)
        # 添加到close表(new,father)
        Closed.append((Min[0], Min[1]))
        state = Min[0]
        moveAndAdd(state, Min[3], dim)


def findState(table, state):
    for i in table:
        if (judgeSame(i[0], state)):
            return i[1]
    return -1


def findAndPrint(Closed, dim):
    outcome = []
    current = rightPos
    while (current != -1):
        if (current != -1):
            outcome.append(current)
        father = findState(Closed, current)
        current = father

    print('搜索次数:', len(Closed) - 1)
    print('搜索路径长度：', len(outcome) - 1)
    for i in range(len(outcome)):
        print('第', i, '步:')
        printState(outcome[len(outcome) - 1 - i], dim)


def main():
    dim = int(input())
    # 输入格式：1 2 3\n4 5 6\n7 8 0\n
    # initState = [[1, 2, 3], [0, 8, 4], [7, 6, 5]]
    # initState=[[1, 2, 3], [7, 8, 4],[ 6, 5, 0]]
    # initState = [[0, 1, 7], [6, 5, 2], [3, 8, 4]]
    # 无解
    # initState = [[2, 1, 3], [8, 0, 4], [7, 6, 5]]
    # initState = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 0, 24]]
    initState = [[0] * dim] * dim
    for i in range(dim):
        initState[i] = input().split(' ')
        initState[i] = [int(j) for j in initState[i]]
    print('初始状态：', initState)
    if (isSolvable(initState, rightPos, dim)):
        A_Star(initState, dim)
        findAndPrint(Closed, dim)
    else:
        print('该问题无解')


if __name__ == '__main__':
    main()
# 函数引用关系绘制
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput
#
# if __name__ == '__main__':
#     graphviz = GraphvizOutput()
#     graphviz.output_file = 'outcome.png'
#     with PyCallGraph(output=graphviz):
#         main()
