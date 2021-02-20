import numpy as np
import csv
import math

tokenized = []
vocabulary = []
labels = []


with open('tokenized_corpus.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        tokenized.append(row)
    #print(len(tokenized))
    
with open('labels.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        labels.append(int(row[0]))


    
for i in range (len(tokenized)):
    for k in range(len(tokenized[i])):
        if tokenized[i][k] not in vocabulary:
            vocabulary.append(tokenized[i][k])

feature_set = np.zeros((len(tokenized),len(vocabulary)))

for i in range (len(tokenized)):
    for k in range(len(vocabulary)):
        count = 0
        if vocabulary[k] in tokenized[i]:
            count = tokenized[i].count(vocabulary[k])
        feature_set[i][k] = count
        

  
np.savetxt("feature_set.csv",feature_set,fmt = '%d',delimiter= ',')


def naive_bayes(feature_set,alpha):
    training_data = feature_set[0:4460]
    test_data = feature_set[4460:]
    
    test_data_labels = labels[4460:]    
    
    N=len(training_data)
    
    
    ham_tetas = []
    spam_tetas = []
    t_hams = []
    t_spams = []
    y_predicts = []
    
    
    spams = []
    for i in range(len(training_data)):
        if labels[i] == 1:
            spams.append(training_data[i])
    hams = []
    for i in range(len(training_data)):
        if labels[i]== 0:
            hams.append(training_data[i])
    
    #T_spam
    def calculate_T_spam(word_j_index):
        T_spam = 0
        for i in range(len(spams)):
            T_spam = T_spam + spams[i][word_j_index].tolist()
        return T_spam    
    
            
    
    #T_ham
    def calculate_T_ham(word_j_index):
        T_ham = 0
        for i in range(len(hams)):
            T_ham = T_ham + hams[i][word_j_index].tolist()
        return T_ham
    
    #Calculate T values
    for j in range(len(feature_set[0])):
        t_hams.append(calculate_T_ham(j))
        t_spams.append(calculate_T_spam(j))
    
    
    
    #N_spam
    N_spam = len(spams)
    #spam_sms prob
    spam_sms = N_spam/N
    
    #N_ham
    N_ham = len(hams)
    #ham_sms prob
    ham_sms = N_ham/N    
    
    
    total_ham = 0
    for j in range(len(feature_set[0])):
        total_ham = total_ham + t_hams[j]
        
    total_spam = 0
    for j in range(len(feature_set[0])):
        total_spam = total_spam + t_spams[j]
    
   
    #Mle spam (laplace)
    def calculate_mle_spam_l(word_j_index,total_spam_t):
        spam_mle = 0
    
        spam_mle = (t_spams[word_j_index]+alpha)/(total_spam_t + alpha*len(feature_set[0]))
        return spam_mle
    
    #Mle ham (laplace)
    def calculate_mle_ham_l(word_j_index,total_ham_t):
        ham_mle = 0
        
        ham_mle = (t_hams[word_j_index] + alpha )/(total_ham_t + alpha*len(feature_set[0]))
        return ham_mle 
            
    
    #laplace
    spam_tetas_l = []
    ham_tetas_l = []
    for j in range(len(feature_set[0])):
        spam_tetas_l.append(calculate_mle_spam_l(j,total_spam))
        ham_tetas_l.append(calculate_mle_ham_l(j,total_ham))
    
    y_predicts = []    
    for i in range(len(test_data)):
        ham_predict_sum = 0
        spam_predict_sum = 0
        for j in range(len(feature_set[0])):  
            if ham_tetas_l[j]== 0 :
                if test_data[i][j]==0:
                    ham_predict_sum = ham_predict_sum + 0
                else:
                    ham_predict_sum = ham_predict_sum + -math.inf
            else:
                ham_predict_sum = ham_predict_sum+test_data[i][j] *math.log(ham_tetas_l[j])
                
            if spam_tetas_l[j]== 0 :
                if test_data[i][j]==0 :
                    spam_predict_sum = spam_predict_sum + 0
                else:
                    spam_predict_sum = spam_predict_sum + -math.inf
            else:
                spam_predict_sum = spam_predict_sum+test_data[i][j] *math.log(spam_tetas_l[j])
                
        ham_predict = math.log(ham_sms) +  ham_predict_sum
        spam_predict = math.log(spam_sms) + spam_predict_sum     
        if ham_predict > spam_predict:
            y_predicts.append(0)
        else:
            y_predicts.append(1)
    
    accuracted_predict = 0
    for i in range(len(y_predicts)):
        if y_predicts[i]==test_data_labels[i] :
            accuracted_predict = accuracted_predict + 1
    accuracy_laplace = [accuracted_predict/len(test_data_labels)]
    return accuracy_laplace



#without laplace

accuracy = naive_bayes(feature_set,0)
np.savetxt('test_accuracy.csv',accuracy,fmt="%f")
#print(accuracy)

#laplace
accuracy_laplace = naive_bayes(feature_set,1)
np.savetxt('test_accuracy_laplace.csv',accuracy_laplace,fmt="%f")
#print(accuracy_laplace)


#Question 3

#New feature_set

vocabulary_r = []
for i in range (len(vocabulary)):
    count = 0
    for k in range(len(feature_set)):        
        count = count + feature_set[k][i]
    if(count >=10):
        vocabulary_r.append( vocabulary[i] )   

new_features = np.zeros((len(tokenized),len(vocabulary_r)))
for i in range (len(tokenized)):
    for k in range(len(vocabulary_r)):
        count = 0
        if vocabulary_r[k] in tokenized[i]:
            count = tokenized[i].count(vocabulary_r[k])
        new_features[i][k]= count

#Q3.1
#forward selection
def forward_selection(features):
    selected_f = []
    temp = []
    is_increasing = True
    max_score = 0

    g = []
    selected_index = 0
    while is_increasing == True:
      current_score = max_score
      for i in range(len(features[0])):
        if i not in selected_f:
          temp = g
          temp.append(i)
          score = naive_bayes(features[:,temp],1)[0]

          if score > max_score:
            max_score = score
            selected_index = i

          temp.pop()
      if current_score == max_score:
            is_increasing = False
      else:
            selected_f.append(selected_index)
            g.append(selected_index)
    np.savetxt('forward_selection.csv',selected_f,fmt ='%d')
    return selected_f

forward_selection(new_features)

#Q3.2
def frequency_selection(features):
  indices = {}
  accuracies= []
  temp = []
  frequencies = np.sum(features[0:4460],axis = 0 )#count of vocabs
  for i in range(len(frequencies)):
    indices[frequencies[i]] = i
  sorted(frequencies,reverse =True)
  
  for i in range(len(features[0])):
    temp.append(indices[frequencies[i]])
    accuracies.append(naive_bayes(features[:,temp],1))
  np.savetxt('frequency_selection.csv',accuracies,fmt ='%f')
frequency_selection(new_features)
 
