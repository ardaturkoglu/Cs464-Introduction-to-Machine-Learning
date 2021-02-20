import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gradient(train_x, train_y, size):
    weights = np.zeros((7, 1))
    z = train_x.dot(weights)
    sigmoid = 1 / (1 + np.exp(-z))
    gradient_weights = np.dot(train_x.T , (train_y - sigmoid)) / size #(x*y)/size
    return  gradient_weights

def train(x,y,l_factor):
    x = np.array(x)
    y = np.array(y)
    beta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    if l_factor != 0:
        beta -= 0.0001*gradient(x,y,len(x)) + l_factor*np.sign(beta)
    return beta


dataset = pd.read_csv("q2_dataset.csv").to_numpy()
np.random.shuffle(dataset)
#Split into 5
split = np.array_split(dataset, 5)
fold1 = split[0]
fold2 = split[1]
fold3 = split[2]
fold4 = split[3]
fold5 = split[4]

test1=fold1[:,[0,1,2,3,4,5,6]]
test2=fold2[:,[0,1,2,3,4,5,6]]
test3=fold3[:,[0,1,2,3,4,5,6]]
test4=fold4[:,[0,1,2,3,4,5,6]]
test5=fold5[:,[0,1,2,3,4,5,6]]

test_label1=fold1[:,[7]]
test_label2=fold2[:,[7]]
test_label3=fold3[:,[7]]
test_label4=fold4[:,[7]]
test_label5=fold5[:,[7]]


train1 = np.delete(split,0,0).reshape(400,8)[:,[0,1,2,3,4,5,6]]
train2 = np.delete(split,1,0).reshape(400,8)[:,[0,1,2,3,4,5,6]]
train3 = np.delete(split,2,0).reshape(400,8)[:,[0,1,2,3,4,5,6]]
train4 = np.delete(split,3,0).reshape(400,8)[:,[0,1,2,3,4,5,6]]
train5 = np.delete(split,4,0).reshape(400,8)[:,[0,1,2,3,4,5,6]]

train_label1= np.delete(split,0,0).reshape(400,8)[:,[7]]
train_label2= np.delete(split,1,0).reshape(400,8)[:,[7]]
train_label3= np.delete(split,2,0).reshape(400,8)[:,[7]]
train_label4= np.delete(split,3,0).reshape(400,8)[:,[7]]
train_label5= np.delete(split,4,0).reshape(400,8)[:,[7]]

#acquiring ß(weights)
beta1= train(train1,train_label1,0)
beta2= train(train2,train_label2,0)
beta3= train(train3,train_label3,0)
beta4= train(train4,train_label4,0)
beta5= train(train5,train_label5,0)

#acquiring y(test predicts) 
predict1_test = test1.dot(beta1)
predict2_test = test2.dot(beta2)
predict3_test = test3.dot(beta3)
predict4_test = test4.dot(beta4)
predict5_test = test5.dot(beta5)
predicts_test = np.array([predict1_test,predict2_test,predict3_test,predict4_test,predict5_test])


#R^2
r2_1 =1 - (np.sum((test_label1-predict1_test)**2)/(np.sum((test_label1-np.mean(predict1_test))**2))) 
r2_2 =1 - (np.sum((test_label2-predict2_test)**2)/(np.sum((test_label2-np.mean(predict2_test))**2))) 
r2_3 =1 - (np.sum((test_label3-predict3_test)**2)/(np.sum((test_label3-np.mean(predict3_test))**2))) 
r2_4 =1 - (np.sum((test_label4-predict4_test)**2)/(np.sum((test_label4-np.mean(predict4_test))**2)))
r2_5 =1 - (np.sum((test_label5-predict5_test)**2)/(np.sum((test_label5-np.mean(predict5_test))**2)))
r2 =[r2_1,r2_2,r2_3,r2_4,r2_5]

#MSE
mse1= np.sum((predict1_test - test_label1)**2) / len(test_label1)
mse2= np.sum((predict2_test - test_label2)**2) / len(test_label2)
mse3= np.sum((predict3_test - test_label3)**2) / len(test_label3)
mse4= np.sum((predict4_test - test_label4)**2) / len(test_label4)
mse5= np.sum((predict5_test - test_label5)**2) / len(test_label5)
mse =[mse1,mse2,mse3,mse4,mse5]

mae1=np.sum(abs(predict1_test-test_label1))/(len(test_label1))
mae2=np.sum(abs(predict2_test-test_label2))/(len(test_label2))
mae3=np.sum(abs(predict3_test-test_label3))/(len(test_label3))
mae4=np.sum(abs(predict4_test-test_label4))/(len(test_label4)) 
mae5=np.sum(abs(predict5_test-test_label5))/(len(test_label5))
mae =[mae1,mae2,mae3,mae4,mae5] 

mape1=(1/len(test_label1))*np.sum(abs((predict1_test-test_label1)/test_label1))
mape2=(1/len(test_label2))*np.sum(abs((predict2_test-test_label2)/test_label2))
mape3=(1/len(test_label3))*np.sum(abs((predict3_test-test_label3)/test_label3))
mape4=(1/len(test_label4))*np.sum(abs((predict4_test-test_label4)/test_label4))
mape5=(1/len(test_label5))*np.sum(abs((predict5_test-test_label5)/test_label5))
mape =[mape1,mape2,mape3,mape4,mape5]

print("Performance Metrics for Q2.2 (Lambda = 0)")
print("R^2 on first fold",r2_1)
print("R^2 on second fold",r2_2)
print("R^2 on third fold",r2_3)
print("R^2 on fourth fold",r2_4)
print("R^2 on fifth fold",r2_5)
print("-----------------------------")
print("Mse(Mean Squared Error) on first fold",mse1)
print("Mse(Mean Squared Error) on second fold",mse2)
print("Mse(Mean Squared Error) on third fold",mse3)
print("Mse(Mean Squared Error) on fourth fold",mse4)
print("Mse(Mean Squared Error) on fifth fold",mse5)
print("-----------------------------")
print("Mae(Mean Absolute Error) on first fold",mae1)
print("Mae(Mean Absolute Error) on second fold",mae2)
print("Mae(Mean Absolute Error) on third fold",mae3) 
print("Mae(Mean Absolute Error) on fourth fold",mae4)
print("Mae(Mean Absolute Error) on fifth fold",mae5)
print("-----------------------------")
print("Mape(Mean Absolute Percentage Error) on first fold",mape1)
print("Mape(Mean Absolute Percentage Error) on second fold",mape2)
print("Mape(Mean Absolute Percentage Error) on third fold",mape3) 
print("Mape(Mean Absolute Percentage Error) on fourth fold",mape4)
print("Mape(Mean Absolute Percentage Error) on fifth fold",mape5)

#Q2.3
#acquiring ß(weights)
beta1= train(train1,train_label1,0.01)
beta2= train(train2,train_label2,0.01)
beta3= train(train3,train_label3,0.01)
beta4= train(train4,train_label4,0.01)
beta5= train(train5,train_label5,0.01)

#acquiring y(test predicts) 
predict1_test = test1.dot(beta1)
predict2_test = test2.dot(beta2)
predict3_test = test3.dot(beta3)
predict4_test = test4.dot(beta4)
predict5_test = test5.dot(beta5)
predicts_test = np.array([predict1_test,predict2_test,predict3_test,predict4_test,predict5_test])


#R^2
r2_1 =1 - (np.sum((test_label1-predict1_test)**2)/(np.sum((test_label1-np.mean(predict1_test))**2))) 
r2_2 =1 - (np.sum((test_label2-predict2_test)**2)/(np.sum((test_label2-np.mean(predict2_test))**2))) 
r2_3 =1 - (np.sum((test_label3-predict3_test)**2)/(np.sum((test_label3-np.mean(predict3_test))**2))) 
r2_4 =1 - (np.sum((test_label4-predict4_test)**2)/(np.sum((test_label4-np.mean(predict4_test))**2)))
r2_5 =1 - (np.sum((test_label5-predict5_test)**2)/(np.sum((test_label5-np.mean(predict5_test))**2)))
r2_l =[r2_1,r2_2,r2_3,r2_4,r2_5]
#MSE
mse1= np.sum((predict1_test - test_label1)**2) / len(test_label1)
mse2= np.sum((predict2_test - test_label2)**2) / len(test_label2)
mse3= np.sum((predict3_test - test_label3)**2) / len(test_label3)
mse4= np.sum((predict4_test - test_label4)**2) / len(test_label4)
mse5= np.sum((predict5_test - test_label5)**2) / len(test_label5)
mse_l =[mse1,mse2,mse3,mse4,mse5]

mae1=np.sum(abs(predict1_test-test_label1))/(len(test_label1))
mae2=np.sum(abs(predict2_test-test_label2))/(len(test_label2))
mae3=np.sum(abs(predict3_test-test_label3))/(len(test_label3))
mae4=np.sum(abs(predict4_test-test_label4))/(len(test_label4)) 
mae5=np.sum(abs(predict5_test-test_label5))/(len(test_label5)) 
mae_l =[mae1,mae2,mae3,mae4,mae5] 

mape1=(1/len(test_label1))*np.sum(abs((predict1_test-test_label1)/test_label1))
mape2=(1/len(test_label2))*np.sum(abs((predict2_test-test_label2)/test_label2))
mape3=(1/len(test_label3))*np.sum(abs((predict3_test-test_label3)/test_label3))
mape4=(1/len(test_label4))*np.sum(abs((predict4_test-test_label4)/test_label4))
mape5=(1/len(test_label5))*np.sum(abs((predict5_test-test_label5)/test_label5))
mape_l =[mape1,mape2,mape3,mape4,mape5]

print("Performance Metrics for Q2.2(Lambda = 0.01,Learning Rate = 0.0001)")
print("R^2 on first fold",r2_1)
print("R^2 on second fold",r2_2)
print("R^2 on third fold",r2_3)
print("R^2 on fourth fold",r2_4)
print("R^2 on fifth fold",r2_5)
print("-----------------------------")
print("Mse(Mean Squared Error) on first fold",mse1)
print("Mse(Mean Squared Error) on second fold",mse2)
print("Mse(Mean Squared Error) on third fold",mse3)
print("Mse(Mean Squared Error) on fourth fold",mse4)
print("Mse(Mean Squared Error) on fifth fold",mse5)
print("-----------------------------")
print("Mae(Mean Absolute Error) on first fold",mae1)
print("Mae(Mean Absolute Error) on second fold",mae2)
print("Mae(Mean Absolute Error) on third fold",mae3) 
print("Mae(Mean Absolute Error) on fourth fold",mae4)
print("Mae(Mean Absolute Error) on fifth fold",mae5)
print("-----------------------------")
print("Mape(Mean Absolute Percentage Error) on first fold",mape1)
print("Mape(Mean Absolute Percentage Error) on second fold",mape2)
print("Mape(Mean Absolute Percentage Error) on third fold",mape3) 
print("Mape(Mean Absolute Percentage Error) on fourth fold",mape4)
print("Mape(Mean Absolute Percentage Error) on fifth fold",mape5)



plt.figure(figsize=(20,10))

plt.subplot(2,2,1)   
plt.plot([1,2,3,4,5],r2,color="r") 
plt.xlabel("Test Folds")
plt.ylabel("R^2")
plt.title("R^2 (Linear Regression)")

plt.subplot(2,2,2)
plt.plot([1,2,3,4,5],r2_l,color="r")
plt.xlabel("Test Folds")
plt.ylabel("R^2")
plt.title("R^2 (Lasso, Lambda = 0.01,Learning Rate = 0.0001)")

plt.show()

plt.figure(figsize=(20,10))

plt.subplot(2,2,1)   
plt.plot([1,2,3,4,5],mse,color="r") 
plt.xlabel("Test Folds")
plt.ylabel("Mse")
plt.title("Mse (Linear Regression)")

plt.subplot(2,2,2)
plt.plot([1,2,3,4,5],mse_l,color="r")
plt.xlabel("Test Folds")
plt.ylabel("Mse")
plt.title("Mse (Lasso, Lambda = 0.01,Learning Rate = 0.0001)")

plt.show()

plt.figure(figsize=(20,10))

plt.subplot(2,2,1)   
plt.plot([1,2,3,4,5],mae,color="r") 
plt.xlabel("Test Folds")
plt.ylabel("Mae")
plt.title("Mae (Linear Regression)")

plt.subplot(2,2,2)
plt.plot([1,2,3,4,5],mae_l,color="r")
plt.xlabel("Test Folds")
plt.ylabel("Mae")
plt.title("Mae (Lasso, Lambda = 0.01,Learning Rate = 0.0001)")

plt.show()

plt.figure(figsize=(20,10))

plt.subplot(2,2,1)   
plt.plot([1,2,3,4,5],mape,color="r") 
plt.xlabel("Test Folds")
plt.ylabel("Mape")
plt.title("Mape (Linear Regression)")

plt.subplot(2,2,2)
plt.plot([1,2,3,4,5],mape_l,color="r")
plt.xlabel("Test Folds")
plt.ylabel("Mape")
plt.title("Mape (Lasso, Lambda = 0.01,Learning Rate = 0.0001)")

plt.show()