from pprint import pprint

import random
from typing import TypeVar, List, Tuple
X = TypeVar('X')  # generic type to represent a data point
Y = TypeVar('Y')  # generic type to represent output variables
Z = TypeVar('Z')

#기존 split_data는 랜덤으로 데이터를 섞기때문에
#저장된 순서대로 데이터를 나누기위해 새로 만드는 함수
def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]                    # Make a shallow copy
    #random.seed(0)
    #random.shuffle(data)             #이부분이 랜덤하게 섞어주는 부분
    cut = int(len(data) * prob)       # Use prob to find a cutoff
    return data[:cut], data[cut:]     # and split the shuffled list there.


def train_test_split(xs: List[X],ys: List[Y],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    # Generate the indices and split them.
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],  # x_train
            [xs[i] for i in test_idxs],   # x_test
            [ys[i] for i in train_idxs],  # y_train
            [ys[i] for i in test_idxs])








#dataframe을 사용하지 못하므로 2차원배열로 불러옴
with open('data2.csv') as file:
    data2 = []
    for line in file.readlines():
        line = line.strip()
        data2.append(line.split(','))


#data2의 전체 데이터 출력
#pprint(data2,width=220)

print('데이터 전체 갯수 : ',len(data2))
fdust_30cm_list = []
ufdust_list = []

for i in range(1,len(data2)):
    fdust_30cm_list.append(data2[i][12]) #1-2에서 새로 추가한 컬럼 fdust+30cm


for i in range(1,len(data2)):
    ufdust_list.append(data2[i][7]) #초미세먼지 컬럼

print('fdust+30cm : ',fdust_30cm_list)
print('ufdust : ',ufdust_list)
fdust_30cm_list = list(map(float, fdust_30cm_list) )
ufdust_list = list(map(int, ufdust_list) )


#from scratch.machine_learning import train_test_split
#이 함수는 위에서 새롭게 재정의함

x_train, x_test, y_train ,y_test = train_test_split(fdust_30cm_list,ufdust_list,0.3)
print('훈련데이터 갯수 :', len(x_train))

from scratch.simple_linear_regression import least_squares_fit, r_squared,error,sum_of_sqerrors

#최소자승법 (훈련데이터에 의해)
alpha , beta = least_squares_fit(x_train,y_train)


#경사하강법
#from scratch.gradient_descent import gradient_step


#R제곱값 구함. (실제 데이터에 의해)
rr = r_squared(alpha,beta,x_test,y_test)
# random으로 데이터 잘랐을때  --> random.seed(0)
# R^2 = 0.80

print('beta : ',beta)
print('R^2: ', rr )

#이거이후에 미세먼지 컬럼으로도 한번 빼보기