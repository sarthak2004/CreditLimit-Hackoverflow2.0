import gradio as gr
import numpy as np
import  requests 
import pandas as pd
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('./flagged/credit_score_predict.csv')

dataset_2 = dataset.drop(['Loan ID', 'Customer ID', 'Months since last delinquent'], axis = 'columns')

dataset_3 = dataset_2.copy()
dataset_3 = dataset_2.dropna()
dataset_3['refined_jobYears'] = dataset_2['Years in current job'].apply( lambda x: (''.join(filter(str.isdigit, str(x)))))
dataset_3['refined_jobYears'] = dataset_3['refined_jobYears'].apply( lambda x: int(x))

dataset_4 = dataset_3.replace(to_replace = ['HaveMortgage'], value = ['Home Mortgage'])

dataset_4.rename(columns = {'Credit Score' : 'Credit_Score','Current Credit Balance' : 'Current_Credit_Balance', 'Maximum Open Credit' : 'Maximum_Open_Credit' ,'Current Loan Amount' : 'Current_Loan_Amount', 'Annual Income' : 'Annual_income', 'Monthly Debt' : 'Monthly_Debt', 'Years of Credit History' : 'Years_of_Credit_History', 'Number of Open Accounts' : 'Number_of_Open_Accounts', 'Number of Credit Problems' : 'Number_of_Credit_Problems', 'Home Ownership' : 'Home_Ownership', 'Tax Liens' : 'Tax_Liens'}, inplace = True)


dataset_5 = dataset_4.replace(to_replace = ['other', 'major_purchase', 'Home Improvements', 'vacation', 'wedding', 'Take a Trip', 'moving', 'small_business'], value = ['Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Other', 'Business Loan'])

dummies = pd.get_dummies(dataset_5.Home_Ownership)
dummies2 = pd.get_dummies(dataset_5.Term)
dataset_6 = pd.concat([dataset_5, dummies, dummies2], axis = 'columns')
dataset_7 = dataset_6.drop(['Years in current job', 'Home_Ownership', 'Purpose', 'Rent', 'Term', 'Short Term'], axis = 'columns')
dataset_7.rename(columns = {'Home Mortgage' : 'Home_Mortgage', 'Own Home' : 'Own_Home', 'Long Term' : 'Long_Term'})

dataset_7.dropna()

X = dataset_7.drop(['Maximum_Open_Credit', 'Number_of_Credit_Problems', 'Long Term'], axis = 'columns')
Y = dataset_7.Maximum_Open_Credit
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 15)
LR = LinearRegression()
LR.fit(X_train, Y_train)

def predict_credit_limit (Ownership, Current_Loan_Amount, Bankruptcies, Tax_Liens, Credit_Score, Annual_Income, Monthly_Debt, Years_of_Credit_History, Number_of_Open_Accounts, Current_Credit_Balance, refined_jobYears):
  ownership_index = np.where(X.columns == Ownership)[0]
  x = np.zeros(len(X.columns))
  x[0] = Current_Loan_Amount
  x[1] = Credit_Score
  x[2] = Annual_Income
  x[3] = Monthly_Debt
  x[4] = Years_of_Credit_History
  x[5] = Number_of_Open_Accounts
  x[6] = Current_Credit_Balance
  x[7] = Bankruptcies
  x[8] = Tax_Liens
  x[9] = refined_jobYears
  
  if ownership_index >= 0:
    x[ownership_index] = 1
  
  return LR.predict([x])[0]









def credit_calculate(Home_owenership,Annual_Income,Credit_Score ,Current_Loan_Amount,Montlhy_Debt,Account_Balance,Number_Of_Open_Accounts,Years_Since_Account_Opening,Tax_liens,Years_Since_Current_Job,Bankruptcies):
    return predict_credit_limit(Home_owenership,int(Current_Loan_Amount),Bankruptcies,Tax_liens ,Credit_Score ,int(Annual_Income),int(Montlhy_Debt),float(Years_Since_Account_Opening),int(Number_Of_Open_Accounts),int(Account_Balance),int(Years_Since_Current_Job))
    




interface = gr.Interface(

    fn = credit_calculate,
    inputs = [gr.Radio(["Own_Home", "Home_Mortgage", "Rent"]),"text", gr.inputs.Slider(300,900,label="Credit Score"),"text", "text","text", "text","text",gr.inputs.Slider(0,10,label="Tax_liens"), "text",gr.inputs.Slider(0,10,label="Bankruptcies")],
    outputs = ["text"]


)

interface.launch()






