# -*- coding: utf-8 -*-
from PIL import Image
import os 
import numpy as np
import matplotlib.pyplot as plt

images = []
im_list = []
x = np.zeros((877,4096,3))
   
os.chdir("van_gogh")
len_van_gogh = os.listdir()

for i in range (1,len(len_van_gogh)+1):
  im = Image.open('Vincent_van_Gogh_' + str(i) +'.jpg')
  if np.asarray(im).shape != (64,64,3):
      im = np.stack((im,)*3, axis=-1)
  im_list.append(np.asarray(im))  
  images.append(np.asarray(im).reshape(4096,3))
for i in range (len(x)):
    x[i] = images[i]


x_1 =x[:,:,0]
x_2=x[:,:,1]
x_3=x[:,:,2]

#center data
#subract means
x_1 = x_1 - np.ones((len(x_1),1))*x_1.mean(axis=0)
x_1 = x_2 - np.ones((len(x_2),1))*x_2.mean(axis=0)
x_1 = x_3 - np.ones((len(x_3),1))*x_2.mean(axis=0)

U1,s1,V1 = np.linalg.svd(x_1)
U2,s2,V2 = np.linalg.svd(x_2)
U3,s3,V3 = np.linalg.svd(x_3)

fig = plt.figure(figsize = (10, 5)) 
# creating the bar plot 

arr=np.arange(1,101)

plt.bar(arr,s1[:100]) 
  
plt.xlabel("Principal Components") 
plt.ylabel("Singular Values") 
plt.title("Top 100 singular values of red channel") 
plt.show() 

fig2 = plt.figure(figsize = (10, 5)) 
plt.bar(arr,s2[:100]) 
  
plt.xlabel("Principal Components") 
plt.ylabel("Singular Values") 
plt.title("Top 100 singular values of green channel") 
plt.show()

fig3 = plt.figure(figsize = (10, 5)) 
plt.bar(arr,s3[:100]) 
  
plt.xlabel("Principal Components") 
plt.ylabel("Singular Values") 
plt.title("Top 100 singular values of blue channel") 
plt.show()
#Pve

eigen_red = np.power(s1,2)
eigen_green= np.power(s2,2)
eigen_blue = np.power(s3,2)
                                        
pve_red = []
pve_green = []
pve_blue = []

for i in eigen_red[:10]:
     pve_red.append((i/sum(eigen_red))*100)
for i in eigen_green[:10]:
     pve_green.append((i/sum(eigen_green))*100)
for i in eigen_blue[:10]:
     pve_blue.append((i/sum(eigen_blue))*100)
        
print("Q1.1 Red channel proportion of variance explained",pve_red)
print("Q1.1 Green channel proportion of variance explained",pve_green)
print("Q1.1 Blue channel proportion of variance explained",pve_blue)

#############################################################
#Q1.2

x1_mean = np.zeros((64,64))
x2_mean = np.zeros((64,64))
x3_mean = np.zeros((64,64))
x1_var = np.zeros((64,64))
x2_var = np.zeros((64,64))
x3_var = np.zeros((64,64))
for i in range(len(im_list)):
    x1_mean += im_list[i][:,:,0]/877
    x2_mean += im_list[i][:,:,1]/877
    x3_mean += im_list[i][:,:,2]/877
for i in range(len(im_list)):
    x1_var += np.power((im_list[i][:,:,0]-x1_mean),2)/877
    x2_var += np.power((im_list[i][:,:,1]-x1_mean),2)/877
    x3_var += np.power((im_list[i][:,:,2]-x1_mean),2)/877
means = np.dstack([x1_mean,x2_mean,x3_mean])
variances = np.dstack([x1_var,x2_var,x3_var])

noised_img = []
for i in range(len(im_list)):
    noised_img.append(im_list[i] + 0.01 * np.random.normal(means, np.sqrt(variances), im_list[i].shape))


for i in range (len(x)):
    x[i] = noised_img[i].reshape(4096,-1)

x_1 =x[:,:,0]
x_2=x[:,:,1]
x_3=x[:,:,2]

#subract means
x_1 = x_1 - np.ones((len(x_1),1))*x_1.mean(axis=0)
x_1 = x_2 - np.ones((len(x_2),1))*x_2.mean(axis=0)
x_1 = x_3 - np.ones((len(x_3),1))*x_2.mean(axis=0)

U1,s1,V1 = np.linalg.svd(x_1)
U2,s2,V2 = np.linalg.svd(x_2)
U3,s3,V3 = np.linalg.svd(x_3)

fig = plt.figure(figsize = (10, 5)) 
# creating the bar plot 

arr=np.arange(1,101)

plt.bar(arr,s1[:100]) 
  
plt.xlabel("Principal Components") 
plt.ylabel("Singular Values") 
plt.title("Top 100 singular values of red channel-noised") 
plt.show() 

fig2 = plt.figure(figsize = (10, 5)) 
plt.bar(arr,s2[:100]) 
  
plt.xlabel("Principal Components-") 
plt.ylabel("Singular Values") 
plt.title("Top 100 singular values of green channel-noised") 
plt.show()

fig3 = plt.figure(figsize = (10, 5)) 
plt.bar(arr,s3[:100]) 
  
plt.xlabel("Principal Components") 
plt.ylabel("Singular Values") 
plt.title("Top 100 singular values of blue channel-noised") 
plt.show()


#PVE
eigen_red = np.power(s1,2)
eigen_green= np.power(s2,2)
eigen_blue = np.power(s3,2)
                                        
pve_red = []
pve_green = []
pve_blue = []

for i in eigen_red[:10]:
     pve_red.append((i/sum(eigen_red))*100)
for i in eigen_green[:10]:
     pve_green.append((i/sum(eigen_green))*100)
for i in eigen_blue[:10]:
     pve_blue.append((i/sum(eigen_blue))*100)
        
print("Q1.2 Noised Image-Red channel proportion of variance explained",pve_red)
print("Q1.2 Noised Image-Green channel proportion of variance explained",pve_green)
print("Q1.2 Noised Image-Blue channel proportion of variance explained",pve_blue)