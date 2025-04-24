# credit_card_fraud_detection
initiallay the data is zipped so unzip that file.
import all the necessary llibraries as immported in the notebook
then read the data using the pandas 
then tried to check the shape of the data
then using the data.head(2) i saw the first 2 rows of data  to analyzze the data
then i tried to drop some columns which are not necessary
then i tried to check if there is any null values in the data there was no null values
then i tried using the one hot encoding becaus data was having too many categorical variables  and convert the categorical variables into binary format so that my model can unnderstand propperly
from sklearn import the trian_test_split
and split the data into 80% for the trianing and 20% for the testing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt import these all the libraries to train the model
to train the model using the decision tree we need to know the what is the min_split and max depth that model should get , to get that lets plot the graph


accuracy_list_train=[]
accuracy_list_test=[]
for min_samples_split in min_samples_split_list:
  model=DecisionTreeClassifier(min_samples_split=min_samples_split,random_state=42).fit(x_train,y_train)
  train_predict=model.predict(x_train)
  test_predict=model.predict(x_test)
  accuracy_train=accuracy_score(train_predict,y_train)
  accuracy_test=accuracy_score(test_predict,y_test)
  accuracy_list_train.append(accuracy_train)
  accuracy_list_test.append(accuracy_test)
plt.title('train and test metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks=range(len(min_samples_split_list)),labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['Train','test'])
with the above code we get the minimum split and that was ranging from the 50 to 100


accuracy_list_train=[]
accuracy_list_test=[]
for max_depth in max_depth_list:
  model=DecisionTreeClassifier(max_depth=max_depth,random_state=42).fit(x_train,y_train)
  train_pedict=model.predict(x_train)
  test_predict=model.predict(x_test)
  accuracy_train=accuracy_score(train_pedict,y_train)
  accuracy_test=accuracy_score(test_predict,y_test)
  accuracy_list_train.append(accuracy_train)
  accuracy_list_test.append(accuracy_test)
plt.title('train and test')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks=range(len(max_depth_list)),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['train','test'])
with this above peice of code  i will get the max depth of the tree and that was between 8 to 16

we got the variables now train the model and after training i got the accuracy of my model was nearly the 99% for both training accuracy nad testing accuracy


