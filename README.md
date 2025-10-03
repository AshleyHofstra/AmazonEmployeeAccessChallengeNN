[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/VLh3GZkS)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=19408365)
## Project 2 CSC 158
	  
A. Pick two datasets, one for a regression problem (NN learns to approximate an function) and dataset for a classification problem (NN learns to predict the class) from kaggle.com links. Please pick datasets with numeric features and with less than 10 input dimension. 
 - Most datasets are saved as a csv file. Use pandas read_csv to read the data into a pandas dataframe (df_data for example).
 - Use df_data.head() to see the first few rows in your table.
 - Use df_data.info() to see description of all columns (number of values, mean, std, etc.)
 - Extract the output column(s).
 	- For regression: y_np = df['y column name'].to_numpy()  - converts it into a numpy array. Check its shape with y_np.shape
  	- For classification: y_np = pd.get_dummies(df['y column name']).to_numpy() - converts labels into one hot encoding (yes/no data into yes column and a no column with 0 and 1 values) (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)
- Extract either all features or a subset of features based on their correlation with the output column.
	- Use the correlation function in pandas (df_data.corr()) (read more here https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)
        - Select the features that have at least 0.2 correlation with the output column.
        - Eliminate features that have a high intercorrelation with another feature (over 0.8)
        - extract the x_np = df_data[[column names separated by comma for the features you want to use]].to_numpy()

B. Normalize the data (subtract the mean and divide by the variance). Then split the data into training and validation (80% training/20% validation). 

 - Use sklearn train/test/split function to divide your data intro training and validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html: y_np_train, y_np_val, x_np_train, x_np_val
 - Use scikit-learn StandardScaler to normalize the **training** data only:https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler (use fit_transform on the x_np_train and y_np_train separately). Then use transform on the x_np_val and y_np_val separately)
 - Convert your numpy arrays  into tensors. Use from_numpy in pytorch: https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
 - You may have to convert the data type of your tensor to float. Use tensor.to(torch.float32).
 - https://pytorch.org/docs/stable/generated/torch.Tensor.to.html

C. Train a NN for each problem. Compute the training loss and the validation loss as in the examples we did in class. Try different architectures: vary the number of hidden layers, number of hidden neurons, learning rate, optimization algorithm (SGD or Adam). Compare the models in terms of training and validation error. Comment on which models gave you better results and why. 

Use **PYTORCH** with Model class as in the example: `iris_nn_pytorch.ipynb` for both classification and regression problems. For a regression problem, the output neuron(s) are linear and the loss function is mean square loss (MSELoss). For a classification problem, the number of neurons in the output layer is equal to the number of classes, the layer is a softmax layer and the loss function is cross-entropy. 

**Class Code/Included files:**

        iris_nn_pytorch.ipynb  = classification iris data with Model class - use this as a starting point to write your code
	restaurant_pred_pytorch.ipynb = regression with the restaurant revenue data with Model class - an example of bad regression results

**Resources**
 - Tutorial pytorch: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html (select other tabs at the top to read about tensors)
 - Numpy arrays intro: https://numpy.org/doc/stable/user/quickstart.html 
 - Pandas (working with tables) intro: https://pandas.pydata.org/docs/user_guide/10min.html 
 - Use the sklearn train/test/split function to divide your data intro training and validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
 - Use scikit-learn to normalize the data: https://scikit-learn.org/stable/modules/preprocessing.html
   
### Submit on the repository: 

I. A report in Comments.md including: 

1. A description of the two problems and datasets you chose;
2. The NN architectures and parameters you tried and the results you obtained; 
3. Comments on the results you obtained and whether learning was successful or not. 

II. Two ipynb files, one for the regression problem and one for the classification problem with results and text comments. 


### Sample problems

UCI repository: https://archive.ics.uci.edu/ml/datasets.php 

https://www.kaggle.com/getting-started/150260

#### a) Regression Problems

How much did it rain :- https://www.kaggle.com/c/how-much-did-it-rain-ii/overview <br>
Inventory Demand:- https://www.kaggle.com/c/grupo-bimbo-inventory-demand <br>
Property Inspection predictiion:- https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction <br>
Restaurant Revenue prediction:- https://www.kaggle.com/c/restaurant-revenue-prediction/data <br>
IMDB Box office Prediction:-https://www.kaggle.com/c/tmdb-box-office-prediction/overview <br>

#### b) Classification problems

Employee Access challenge :- https://www.kaggle.com/c/amazon-employee-access-challenge/overview <br>
Titanic :- https://www.kaggle.com/c/titanic <br>
San Francisco crime:- https://www.kaggle.com/c/sf-crime <br>
Customer satisfcation:-https://www.kaggle.com/c/santander-customer-satisfaction <br>
Trip type classification:- https://www.kaggle.com/c/walmart-recruiting-trip-type-classification <br>
Categorize cusine:- https://www.kaggle.com/c/whats-cooking <br>
