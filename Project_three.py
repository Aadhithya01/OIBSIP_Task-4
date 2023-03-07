import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# TfidfVectorizer is used here because strings cant directly be classified, so it has to be converted to numerical form
# Logistic regression is used because whenever a 2 way selection is involved it is better to use LogisticRegression

data = pd.read_csv('spam.csv', encoding='latin-1')

m_data = data.where((pd.notnull(data)),'')

# Removing unwanted columns

m_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
m_data.rename(columns={'v1':'Type','v2':'Mail Content'},inplace =True)

print(m_data.duplicated().sum())

m_data = m_data.drop_duplicates(keep="first")

# Converting the ham and spam string to a numerical value inorder to be analysed

m_data.loc[m_data['Type'] == 'spam', 'Type',] = 0
m_data.loc[m_data['Type'] == 'ham', 'Type',] = 1

X = m_data['Mail Content']

Y = m_data['Type']

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_Train_features = feature_extraction.fit_transform(X_Train)
X_Test_features = feature_extraction.transform(X_Test)

Y_Train = Y_Train.astype('int')
Y_Test = Y_Test.astype('int')

# Using Logistic Regression

lr = LogisticRegression()
lr.fit(X_Train_features,Y_Train)
y_pred_train = lr.predict(X_Train_features)
accuracy = accuracy_score(Y_Train,y_pred_train)
print(accuracy)


y_pred_test = lr.predict(X_Test_features)
accuracy1 = accuracy_score(Y_Test,y_pred_test)
print(accuracy1)
# Plotting the points and representing the regression line
plt.scatter(Y_Test, y_pred_test)

# Add the regression line
plt.plot(Y_Test, Y_Test, color='red')

# Set the labels and title
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Results')
plt.show()

# Plotting a pie chart

plt.pie(m_data['Type'].value_counts(), labels = ['ham','spam'],autopct="%0.2f")
plt.show()

# Calculating a r2_score which is used to say how well the regression line is present.
# Value nearing to 1 then it is a best fit
r2 = r2_score(Y_Test, y_pred_test)
print(r2)

input_mail = ["watcha doing?"]

input_data_features = feature_extraction.transform(input_mail)

prediction = lr.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')


