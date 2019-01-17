from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import copy
from matplotlib import pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/")

mnist_images = mnist.train.images[:100]
mnist_labels = mnist.train.labels[:100]

num_Of_Pattern_Each_Digit = 5  # 50 digit 1 and 50 digit 5 patterns

def create_weight_matrix(input): #input multiple string
    n = len(input[0])
    p = len(input)
    weight_matrix = np.zeros((n,n))
    for i in range(0,n-1):
        for j in range(i+1,n):
            sum = 0
            for k in range(0,p):
                if(input[k][i]==input[k][j]):
                    sum = sum+1
                else:
                    sum = sum-1
            weight_matrix[i][j] = sum
            weight_matrix[j][i] = sum
    return weight_matrix

# def create_weight_matrix(input):
#     neu = len(input[0])
#     p = len(input)
#     weight_matrix = np.zeros((neu,neu))
#     for inp in input:
#         v = np.zeros((neu,neu))
#         def h(i, j, w, e, n):
#             h_ij = sum([w[i][k] * e[k] for k in range(1, n) if k != i and k != j])
#             return h_ij
#             print("learning")
#         for i in range(0,neu-1):
#             print(i)
#             for j in range(i+1,neu):
#                 if i == j:
#                     continue
#                 v[i][j] = weight_matrix[i][j] + (1.0/neu)*inp[i]*inp[j] - (1.0/neu)*inp[i]*h(j,i,weight_matrix,inp,neu) -(1.0/neu)*h(i,j,weight_matrix,inp,neu)*inp[j]
#                 v[j][i] = v[i][j]
#         weight_matrix = v
#
#
#     return weight_matrix


def process(inputS): #input one string at a time
    count = 0
    input = copy.deepcopy(inputS)
    while(True):
        flag = True
        randNum = list(range(0,len(input)))
        random.shuffle(randNum)
        count += 1
        for j in range(0,len(input)):
            index = randNum[j]   # pick a random bit
            temp = copy.deepcopy(input)
            sum = 0
            for i in range(len(input)):
                sum += m[index][i] * input[i]
            if sum >= 0:
                input[index] = 1
            else:
                input[index] = 0
            if (input[index]!=temp[index]):
                flag = False
            temp = copy.deepcopy(input)
        if flag==True or count == 20:
            return input;


def draw_patterns(start, results):
    print(start)
    print(results)
    print("printing")
    for i in range(num_Of_Pattern_Each_Digit):
        print(i)
        # plt.subplot(10,10,(i+1)*2-1)

        plt.imshow(start[i])
        plt.show()
        # plt.subplot(10,10,(i+1)*2)
        plt.imshow(results[i])
        plt.show()




# Subgroup value 1 and 5

new_image = np.matrix([[]])
new_label = []
new_image_5 = np.matrix([[]])
new_label_5 = []

for i in range(len(mnist_labels)):
    if(mnist_labels[i]==5):
        new_label_5.append(mnist_labels[i])
        if (new_image_5.size == 0):
            new_image_5 = np.matrix(mnist_images[i,:])
        new_image_5 = np.vstack([new_image_5, mnist_images[i,:]])
    if(mnist_labels[i]==1):
        new_label.append(mnist_labels[i])
        if (new_image.size == 0):
            new_image = np.matrix(mnist_images[i,:])
        new_image = np.vstack([new_image, mnist_images[i,:]])

input_data = np.asarray(new_image)
input_data_5 = np.asarray(new_image_5)

input_data = [[1 if p>0 else -1 for p in v] for v in input_data]
input_data_5 = [[1 if p>0 else -1 for p in v] for v in input_data_5]

test = np.array([input_data[5],input_data_5[5]])

m = create_weight_matrix(test)

result = []
print("testing pattern: ")
for i in range(num_Of_Pattern_Each_Digit):
    print(i)
    result.append(process(input_data[i]))
    result.append(process(input_data_5[i]))


for i in range(0, len(result)):
    result[i] = np.asarray(result[i])
    result[i].shape = (28,28)

start = []
for i in range(0, num_Of_Pattern_Each_Digit):
    temp = input_data[i]
    temp = np.asarray(temp)
    temp.shape = (28,28)
    start.append(temp)
    temp = input_data_5[i]
    temp = np.asarray(temp)
    temp.shape = (28,28)
    start.append(temp)

draw_patterns(start,result)
