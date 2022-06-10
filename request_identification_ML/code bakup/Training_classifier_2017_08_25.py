
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')#replace ignore with default for enabling the warning
import gc
import os
import xlsxwriter
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from request_identification.utilities.my_XL_Cls import XL_Results_writing
from request_identification.utilities.my_progressBar import My_progressBar
from request_identification.utilities.my_save_load_model import Save_Load_Model
import pandas as pd
import time
import datetime

class Training_Classifiers:
  
  def cal_accuracy(self, y_test, y_pred_class, clf_name):
    # conf_metrics = metrics.confusion_matrix(y_test, y_pred_class)
    acc = metrics.accuracy_score(y_test, y_pred_class) * 100
    prec = metrics.precision_score(y_test, y_pred_class, average='weighted') * 100
    recal = metrics.recall_score(y_test, y_pred_class, average="weighted") * 100
    f1 = metrics.f1_score(y_test, y_pred_class, average='weighted') * 100
    # if clf_name == "Random Forest":
#     print(["{0:.2f}".format(acc), "{0:.2f}".format(prec), "{0:.2f}".format(recal), "{0:.2f}".format(f1)])
    # saveData_cross_check(y_test, y_pred_class, clf_name)
    return ["{0:.2f}".format(acc), "{0:.2f}".format(prec), "{0:.2f}".format(recal), "{0:.2f}".format(f1)]


  def apply_SVM_KF(self, X_train_dtm, X_test_dtm, y_train):
    clf = svm.SVC()
    clf.fit(X_train_dtm, y_train)
    y_pred_class = clf.predict(X_test_dtm)
    # print("SVM completed")
    return y_pred_class


  def apply_Logistic_KF(self, X_train_dtm, X_test_dtm, y_train):
    logreg = LogisticRegression()
    logreg.fit(X_train_dtm, y_train)
    y_pred_class = logreg.predict(X_test_dtm)
    # print("LR completed")
    return y_pred_class

  def apply_RandomForest_KF(self, X_train_dtm, X_test_dtm, y_train):
    # clf_randomForest = RandomForestClassifier(n_estimators=382, criterion='entropy', max_features=116, max_depth=33, min_samples_split=5, min_samples_leaf=1 )
    clf_randomForest = RandomForestClassifier()
    clf_randomForest.fit(X_train_dtm, y_train)
    y_pred_class = clf_randomForest.predict(X_test_dtm)
    # print("RF completed")
    return y_pred_class

  def apply_NaiveBayes_KF(self, X_train_dtm, X_test_dtm, y_train):
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    # path = 'Classifiers/' + 'NB' + '.pkl'
    # with open(path, 'wb') as f:
    #   pickle.dump(nb, f)
    # print("NB completed")
    return y_pred_class

  def apply_GradientBoostingClf_KF(self, X_train_dtm, X_test_dtm, y_train):
    gb = GradientBoostingClassifier()
    gb.fit(X_train_dtm, y_train)
    y_pred_class = gb.predict(X_test_dtm)
    # print("GB completed")
    return y_pred_class

  def apply_MLP_KF(self, X_train_dtm, X_test_dtm, y_train):
    clf = MLPClassifier(solver='sgd')
    clf.fit(X_train_dtm, y_train)
    y_pred_class = clf.predict(X_test_dtm)
    # print("MLP completed")
    return y_pred_class
  # unit test OKKKK

  def ls_ToDf(self ,ls1 , ls2, ls3, ls4, ls5, ls6):
    '''
    -convert list type into DataFrame and add columns names for creating labelled table of dataframe type
    + takes six lists of arguments and convert it Dataframe row wise order
    '''
    ls_digit1 = list(map(float, ls1))
    ls_digit2 = list(map(float, ls2))
    ls_digit3 = list(map(float, ls3))
    ls_digit4 = list(map(float, ls4))
    ls_digit5 = list(map(float, ls5))
    ls_digit6 = list(map(float, ls6))
    res = list()
    res.append(ls_digit1)
    res.append(ls_digit2)
    res.append(ls_digit3)
    res.append(ls_digit4)
    res.append(ls_digit5)
    res.append(ls_digit6)
    df = pd.DataFrame(res, columns=['Accuracy', 'Precision', 'Recall', 'F1-Measure'])
    df['Classifier']=['Naive Bayes', 'Logisitic Regression', 'SVM', 'Random Forest', 'Gradient Boosting', 'NLP']
    df = df[['Classifier','Accuracy', 'Precision', 'Recall', 'F1-Measure']]
    df.set_index('Classifier', inplace = True)
    return df

  def stratified_cv(self, X, y, uniSpecs, n_splits=5, shuffle=True):
    stratified_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    X_train_dtm = X
    X_train_dtm = X_train_dtm.toarray()
    y_pred_NB = y.copy()
    y_pred_LogReg = y.copy()
    y_pred_RForest= y.copy()
    y_pred_gbClf = y.copy()
    y_pred_SVM = y.copy()
    y_pred_MLP = y.copy()
    num_Clf = 6 #this is equal to number of classifiers used for testing
    iteration = 0
    for train_index, test_index in stratified_k_fold.split(X_train_dtm, y):
      message = "\n\nRunning "+str(iteration+1)+" out of "+str(n_splits)+' fold(s):'+ "for "+ uniSpecs
      pBar = My_progressBar(message,num_Clf)
      start_time = time.time()
      X_train, X_test = X_train_dtm[train_index], X_train_dtm[test_index]
      y_train = y[train_index]
      pBar.call_to_progress(start_time)
      start_time = time.time()
      y_pred_NB[test_index] = self.apply_NaiveBayes_KF(X_train, X_test, y_train)
      pBar.call_to_progress(start_time)
      start_time = time.time()
      y_pred_LogReg[test_index] = self.apply_Logistic_KF(X_train, X_test, y_train)
      pBar.call_to_progress(start_time)
      start_time = time.time()
      y_pred_SVM[test_index] = self.apply_SVM_KF(X_train, X_test, y_train)
      pBar.call_to_progress(start_time)
      start_time = time.time()
      y_pred_RForest[test_index] = self.apply_RandomForest_KF(X_train, X_test, y_train)
      pBar.call_to_progress(start_time)
      start_time = time.time()
      y_pred_gbClf[test_index] = self.apply_GradientBoostingClf_KF(X_train, X_test, y_train)
      pBar.call_to_progress(start_time)
      start_time = time.time()
      y_pred_MLP[test_index] = self.apply_MLP_KF(X_train, X_test, y_train)
      iteration =  iteration + 1
      pBar.call_to_progress(start_time)
      start_time = time.time()
      pBar = ""
      gc.collect()
  #     print(str(iteration) + " fold completed")
    # temp = pd.DataFrame([y, y_pred_NB], columns=['actual', 'predicted'])
    # print(temp.head())
    score_NB = self.cal_accuracy(y, y_pred_NB, "NB")
    score_LogReg = self.cal_accuracy(y, y_pred_LogReg, "Logisict Regression")
    score_SVM = self.cal_accuracy(y, y_pred_SVM, "SVM")
    score_RForest = self.cal_accuracy(y, y_pred_RForest, "Random Forest")
    score_GBClf = self.cal_accuracy(y, y_pred_gbClf, "GB Classifier")
    score_MLP = self.cal_accuracy(y, y_pred_MLP, "MLP Classifier")
    # return score_NB, X_train_dtm.shape
  #   dim[1] = X_train_dtm.shape
    df = self.ls_ToDf(score_NB, score_LogReg, score_SVM, score_RForest, score_GBClf, score_MLP)
  #   xl.save_resultsToExcel(df1, label, label, dim[1])
    # return score_NB, score_LogReg, score_SVM, score_RForest, score_GBClf ,score_MLP, X_train_dtm.shape
    return df

  def list_str(self, list1):
    '''Def: convert list to string by joing list element using _
    Args: pass python list as an argument
    Ret: retrun resultant string'''
    return '_'.join(list1)

  def check_file(self, fname, path = ''):
    if path == '':
      print('please provide path to the folder:')
      exit()
    created_at = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    temp = path + created_at +'_'+ fname
    path =temp + '.xlsx'
    while(os.path.isfile(path) == True):
      old = path
      temp +=' copy'
      path =temp +'.xlsx'
    workbook = xlsxwriter.Workbook(path)
    workbook.close()
    return path

  def extractInfo_specs(self, st, req):
    '''Def: fetch required info from list
    Args: pass list and req arguments where req contained name of the required info to be fetched from list
    Ret: return required info'''
    st = st.split('_')
    if req == 'sheet_name':
      info = st[0]
    elif req == 'file_name':
      info = self.list_str(st[1:])
    elif req == 'freq':
      info = st[-1].split('-')[1]
    return info


  def training_clfs(self, encap_res, y, k_fold = 2):
    for key,value in encap_res.items():
      if key == 'unigrams':
        uniSpecs = value['specs']
        uniTrainDtm = value['dtm']
      elif key == 'uniBigrams':
        uniBiSpecs = value['specs']
        uniBiTrainDtm = value['dtm']
      elif key == 'uniBiTrigrams':
        uniBiTriSpecs = value['specs']
        uniBiTriTrainDtm = value['dtm']
      elif key == 'bigrams':
        biSpecs = value['specs']
        biTrainDtm = value['dtm']
      elif key == 'biTrigrams':
        biTriSpecs = value['specs']
        biTriTrainDtm = value['dtm']
      elif key == 'trigrams':
        triSpecs = value['specs']
        triTrainDtm = value['dtm']
    # checking for file
    resultFileName = self.extractInfo_specs(uniSpecs, 'file_name')


    # pathToClfUniBi = self.check_file(uniBiSpecs, pathToClfFolder)
    # pathToClfUniBiTri = self.check_file(uniBiTriSpecs, pathToClfFolder)
    # pathToClfBi = self.check_file(biSpecs, pathToClfFolder)
    # pathToClfBiTri = self.check_file(biTriSpecs, pathToClfFolder)
    # pathToClfTri = self.check_file(triSpecs, pathToClfFolder)

    pathToResFolder = '../result_phase2_completeDataset/'
    pathTofileName = self.check_file(resultFileName, pathToResFolder)
    xl = XL_Results_writing(pathTofileName)
    # unigrams
    df = self.stratified_cv(uniTrainDtm, y, uniSpecs, n_splits=k_fold)
    xl.save_resultsToExcel(df, self.extractInfo_specs(uniSpecs, 'sheet_name'),
                           uniSpecs, self.extractInfo_specs(uniSpecs, 'freq'))
    # uniBigrams
    df = self.stratified_cv(uniBiTrainDtm, y, uniBiSpecs, n_splits=k_fold)
    xl.save_resultsToExcel(df, self.extractInfo_specs(uniBiSpecs, 'sheet_name')
                           , uniBiSpecs, self.extractInfo_specs(uniBiSpecs, 'freq'))
    # uniBiTrigrams
    df = self.stratified_cv(uniBiTriTrainDtm, y, uniBiTriSpecs, n_splits=k_fold)
    xl.save_resultsToExcel(df, self.extractInfo_specs(uniBiTriSpecs, 'sheet_name')
                           , uniBiTriSpecs, self.extractInfo_specs(uniBiTriSpecs, 'freq'))
    # bigrams
    df = self.stratified_cv(biTrainDtm, y, biSpecs, n_splits=k_fold)
    xl.save_resultsToExcel(df, self.extractInfo_specs(biSpecs, 'sheet_name')
                           , biSpecs, self.extractInfo_specs(biSpecs, 'freq'))
    # biTrigrams
    df = self.stratified_cv(biTriTrainDtm, y, biTriSpecs, n_splits=k_fold)
    xl.save_resultsToExcel(df, self.extractInfo_specs(biTriSpecs, 'sheet_name')
                           , biTriSpecs, self.extractInfo_specs(biTriSpecs, 'freq'))
    # trigrams
    df = self.stratified_cv(triTrainDtm, y, triSpecs, n_splits=k_fold)
    xl.save_resultsToExcel(df, self.extractInfo_specs(triSpecs, 'sheet_name')
                           , triSpecs, self.extractInfo_specs(triSpecs, 'freq'))

    xl.generate_resultantWorkSheet('Accuracy', 'Accuracy')
    xl.generate_resultantWorkSheet('Precision', 'Precision')
    xl.generate_resultantWorkSheet('Recall', 'Recall')
    xl.generate_resultantWorkSheet('F1-Measure', 'F1-Measure')
    
script_start = time.time()


# tc = Training_Classifiers()
# tc.training_clfs(features, y, k_fold = 2)

print("\nscritp completed")
Total_time = time.time() - script_start
print("\nTotal time for script completion :" + str(datetime.timedelta(seconds=int(Total_time))))

