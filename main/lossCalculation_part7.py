import math

# loss calculation is done using negative log formula i.e e ** x = b

softmax_output = [0.2,0.4,8]
lable_output = [1,0,0]

loss = -(math.log(softmax_output[0]*lable_output[0]+softmax_output[1]*lable_output[1]+softmax_output[2]*lable_output[2]))

print(loss)

# # above can be written in below form as well
loss = -(math.log(softmax_output[0]*lable_output[0])) # because other value is being multiply by 0 which equals to zero
print(loss)