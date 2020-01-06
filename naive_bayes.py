import csv
import pandas as pd
from collections import Counter, defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import math
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import utils
from functools import reduce


filename = 'diabetes.csv'
diabetes = pd.read_csv('diabetes.csv')
diabetes = diabetes.sample(frac=1).reset_index(drop=True)

glass = pd.read_csv('glass.csv')
glass = glass.sample(frac=1).reset_index(drop=True)
# print(glass)

wine = pd.read_csv('wine.csv', names=['Wine','Alcohol','Malic.acid','Ash','Acl','Mg','Phenols','Flavanoids','Nonflavanoid.phenols','Proanth','Color.int','Hue','OD','Proline'])
wine = wine.sample(frac=1).reset_index(drop=True)
new_names = names=['Proline','Alcohol','Malic.acid','Ash','Acl','Mg','Phenols','Flavanoids','Nonflavanoid.phenols','Proanth','Color.int','Hue','OD','Wine']
wine = wine[new_names]


# x_data = [list(row[:-1]) for index, row in diabetes.iterrows()]
# x_data_by_col = [list(diabetes[col]) for col in diabetes.columns[:-1]]
# y_data = list(diabetes[diabetes.columns[-1]])

def get_data(dataframe):
    x_data = np.array([list(row[:-1]) for index, row in dataframe.iterrows()])
    y_data = np.array(list(dataframe[dataframe.columns[-1]]))
    return x_data,y_data

def print_stats_df(df):
    class_col = df.columns[-1]
    print(df.groupby(class_col).agg(['count']))
    print("total", len(df))

# print_stats_df(diabetes)



def calc_stats(true_labels, pred_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))  
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0)) 
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1)) 

    precision = (TP + 1) / (TP + FP + 1*2)
    recall = (TP + 1) / (TP + FN + 1*2)
    f1_score = 2 * (precision * recall / (precision + recall))
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1': f1_score
    }



def f1_multi(true_labels, pred_labels):
    unique = np.unique(np.concatenate([true_labels,pred_labels]))
    k = unique.size
    stats_per_class = map(lambda cl: calc_stats(true_labels == cl, pred_labels == cl), unique)
    prec, rec = reduce(lambda acc, stats: (acc[0] + stats['precision'], acc[1] + stats['recall']), stats_per_class, (0, 0))
    prec /= k
    rec /= k
    return 2 * (prec*rec) / (prec + rec)


def cross_validation(dataframe, folds, bins_num, discret_method, acc_method):

    fold_len = len(dataframe)/folds
    divided = [dataframe[int(i*fold_len):int((i+1)*fold_len)] for i in range(0,folds)]
    results = []
    for i in range(folds):
        test_data = divided[i]
        test_x,test_y = test_data[test_data.columns[:-1]], test_data[test_data.columns[-1]]
        test_y = np.array(test_y)
        train_data = divided[:i] + divided[i+1:]
        
        train_data = pd.concat(train_data)
        bins = [bins_num for _ in range(len(dataframe.columns)-1)]
        model = create_model(train_data, bins, discret_method)
        probs = p_y_x_nb(model, test_data)
        pred_labels = get_labels(probs)
        # print(pred_labels)
        # if (folds==10 and i==0):
        #     plot_confusion_matrix(test_y,pred_labels,dataframe.columns[:-1],'Discretisation')

        if (acc_method == 'accuracy'):
            results.append(accuracy(test_y,pred_labels)*100)
        if (acc_method == 'fmulti'):
            results.append(f1_multi(test_y,pred_labels))
        else:
            results.append(f_score(test_y,pred_labels))

    return sum(results) / folds

def conf_matr(data, bins_num, discret):
    bins = [bins_num for _ in range(len(data.columns)-1)]
    msk = np.random.rand(len(data)) < 0.8
    train = data[msk]
    test = data[~msk]
    test_x,test_y = test[test.columns[:-1]], test[test.columns[-1]]
    model = create_model(train, bins, discret)
    probs = p_y_x_nb(model, test)
    pred_labels = get_labels(probs)
    plot_confusion_matrix(test_y,pred_labels,range(1,(data.iloc[:,-1]).unique))

def cross_validation_normal(dataframe, folds, bins_num, discret_method, acc_method):
    fold_len = len(dataframe)/folds
    divided = [dataframe[int(i*fold_len):int((i+1)*fold_len)] for i in range(0,folds)]
    results = []
    for i in range(folds):
        test_data = divided[i]
        test_x,test_y = test_data[test_data.columns[:-1]], test_data[test_data.columns[-1]]
        test_y = np.array(test_y)
        train_data = divided[:i] + divided[i+1:]
        train_data_joined = pd.concat(train_data)

        pred_labels = getPredictions(summarize_by_class(train_data_joined),test_data)
        print("Normal Pred", pred_labels)
        print("Normal True",test_y)
        if (acc_method == 'accuracy'):
            results.append(accuracy(test_y,pred_labels)*100)
        if (acc_method == 'fmulti'):
            results.append(f1_multi(test_y,pred_labels))
        else:
            results.append(f_score(test_y,pred_labels))

        # if (folds==10 and i==0):
        #     plot_confusion_matrix(test_y,pred_labels,dataframe.columns[:-1],'Normal Distribution')

    return sum(results) / folds

# The folds are made by preserving the percentage of samples for each class.
def stratified_cross_validation(dataframe, folds, bins_num, discret_method, acc_method):
    results = []
    x_data,y_data = get_data(dataframe)
    skf = StratifiedKFold(n_splits=folds,random_state=None, shuffle=False)
    skf.get_n_splits(x_data, y_data)
    bins = [bins_num for _ in range(len(dataframe.columns)-1)]
    # StratifiedKFold(n_splits=folds, random_state=None, shuffle=False)
    for train_index, test_index in skf.split(x_data, y_data):
        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        
        train_data = pd.DataFrame(X_train)
        train_data['class'] = y_train
        test_data = pd.DataFrame(X_test)
        test_data['class'] = y_test

        model = create_model(train_data,bins,discret_method)
        probs = p_y_x_nb(model, test_data)
        pred_labels = get_labels(probs)
        if (acc_method == 'accuracy'):
            results.append(accuracy(y_test,pred_labels)*100)
        if (acc_method == 'fmulti'):
            results.append(f1_multi(test_y,pred_labels))
        else:
            results.append(f_score(y_test,pred_labels))
    return sum(results)/folds


def print_min_max_vals(data):
    for col in list(diabetes):
        print(col, 'max: {}'.format(max(list(diabetes[col]))), 'min: {}'.format(min(list(diabetes[col]))))

def preprocess_dataframe(dataframe):
    for j in dataframe.columns[:-1]:
        mean = dataframe[j].mean()
        dataframe[j] = dataframe[j].replace(0,mean)
    return dataframe

wine = preprocess_dataframe(wine)
glass = preprocess_dataframe(glass)
# print(wine)
# zaleta bo kazda kategoria ma kogos w srodku
def discretise_col_len(data_col, bins_num):
    # print(bins_num)
    l = len(data_col)
    samples_in_bin = int(np.ceil(l/bins_num))
    data_col_copy = np.sort(np.array(data_col.copy()))
    bins = data_col_copy[samples_in_bin::samples_in_bin]
    x = np.array(data_col)
    return (np.digitize(x,bins,right=True), bins)

# w równych odległościach od siebie ustawiamy bins
def discretise_col_random(data_col, bins_num):
    maxx = max(data_col)
    minn = min(data_col)
    bins = np.linspace(minn, maxx, bins_num)[1:]
    x = np.array(data_col)
    return (np.digitize(x,bins,right=True), bins)

from sklearn import preprocessing

def discretise_col_entropy(df,data_col,colname,y_data,depth):
    tree_model = DecisionTreeClassifier(max_depth=depth)
    # tree_model = DecisionTreeRegressor(max_depth=depth)
    # print (df[data_col].to_frame())

    lab_enc = preprocessing.LabelEncoder()
    # data = lab_enc.fit_transform(df[data_col].to_frame())
    x_data = data_col.to_frame()

    tree_model.fit(x_data,y_data)
    df['tree'] = tree_model.predict_proba(data_col.to_frame())[:,1]
    bins = list(df.groupby(['tree'])[colname].min())
    bins.sort()
    return (np.digitize(data_col,bins,right=True),bins)

def discretise_dataframe(dataframe,method,bins_array=None):
    if method not in ['len', 'random', 'entropy']:
        raise ValueError("Valid types of discretise method are: len,random,entropy")
    if bins_array is not None:
        if len(bins_array)!=len(dataframe.columns)-1:
            raise ValueError("Length of bins array (%d) must be equal to nr of features (%d)" % (len(bins_array),len(dataframe.columns)-1))
    bins_by_feature = []
    
    discretised_cols = dataframe.copy()
    for bins_num, col in zip(bins_array, list(dataframe.columns[:-1])):
        # print(dataframe[dataframe.columns[-1]])
        if method=='len':
            disco,bins = discretise_col_len(dataframe[col],bins_num)
        if method=='random':
            disco,bins = discretise_col_random(dataframe[col],bins_num)
        if method=='entropy':
            depth = int(math.log(bins_num-1,2))
            disco,bins = discretise_col_entropy(dataframe.copy(),dataframe[col],col,dataframe[dataframe.columns[-1]],depth)
        
        discretised_cols[col] = disco
        bins_by_feature.append(bins)
        # print(bins_by_feature)
    return (discretised_cols, bins_by_feature)

# print(diabetes.dtypes)
# print(discretise_dataframe(diabetes,'entropy',[9,9,9,9,9,9,9,9]))
# 3,5,9,17
# col, bins = discretise_col_entropy(diabetes, diabetes['BloodPressure'],'BloodPressure',diabetes['Outcome'],2)
# print(bins)

# col, bins = discretise_col_len(diabetes['BloodPressure'],9)
# print(bins)

# col, bins = discretise_col_random(diabetes['BloodPressure'],9)
# print(bins)
# for idx, i in enumerate(discretise_column(diabetes['Age'],bins)):
#     print(idx, i)

def count_apriori_prob(ytrain):
    labels = np.unique(ytrain)
    result = [sum(y == label for y in ytrain) / len(ytrain) for label in labels]
    return result

# count_apriori_prob(y_data)

def estimate_p_x_y_nb(xtrain, ytrain, a, b):
    result = np.zeros((len(ytrain),len(xtrain[0])))
    for yidx in range(len(ytrain)):
        for xidx, x in enumerate(xtrain[yidx]):
            label = ytrain[yidx]
            label_indexes = np.where(y_data == label)
            data_col = xtrain.transpose()[xidx]
            different_values_per_col = len(data_col[label_indexes[0].tolist()])
            curr_value_per_col = len(np.where(data_col[label_indexes[0]] == x)[0])
            result[yidx][xidx]=curr_value_per_col/different_values_per_col 
    return (result)

def count(data,colname,category):
    condition = (data[colname] == category)
    return len(data[condition])


# create model, bins_num
# discretise zwroc dane i biny, potem uzyc do dyskretyzacji
# a priori
# prob matrix - lista list list
# model gotowy
# zwracaj - bins, potem uzyc do dyskretyzacji, a prori i prob matrix - dict

def estimate_p_x_y_nb(dataframe, num_bins):
    a = 1
    probabilities=[]
    label_col = dataframe.columns[-1]
    
    for label in range(len(dataframe[label_col].unique())):
        condition = (dataframe[label_col] == label)
        data_with_label = dataframe[condition]
        label_list = []
        for feature_idx, feature in enumerate(dataframe.columns[:-1]):
            feature_list = []
            for category in range(num_bins[feature_idx]):
                prob = (count(data_with_label,feature,category) + a) / (len(data_with_label) + a*num_bins[feature_idx])
                feature_list.append(prob)
            label_list.append(feature_list)
        probabilities.append(label_list)
    return probabilities

# print(estimate_p_x_y_nb(discretised))

def create_model(data, bins, discretise_method):
    data,bins_by_feature = discretise_dataframe(data,discretise_method,bins)
    x_train, y_train = get_data(data)
    apriori = count_apriori_prob(y_train)
    probability_matrix = estimate_p_x_y_nb(data, bins)
    return {'bins':bins_by_feature,'apriori':apriori,'aposteriori':probability_matrix}

def discretise_data(data,bins_array):
    data = data.copy()
    for col,bins in zip(data.columns,bins_array):
        data[col] = np.digitize(data[col],bins,right=True)
    return data

def discretise_column(col,bins):    
    return np.digitize(col,bins,right=True)

def p_y_xrow(row,aposteriori,apriori):
    res = np.zeros(len(apriori))
    for label in range (len(apriori)):
        prob = [aposteriori[label][feature_idx][category] for feature_idx, category in enumerate(row)]
        res[label] = np.prod(prob)*apriori[label]
    return res

def p_y_x_nb(model, x):
    data = discretise_data(x,model['bins'])
    apriori = model['apriori']
    aposteriori = model['aposteriori']
    p_y_x = np.zeros((len(data),len(apriori)))

    for i in range(len(data)):
        row = data.iloc[i,:-1]
        p_y_x[i,:] = p_y_xrow(row, aposteriori,apriori)
    return p_y_x

def get_labels(probs):
    return np.argmax(probs, axis=1)

def accuracy(labels, pred_labels):
    total = len(labels)
    correct = np.sum(pred_labels == labels)
    return correct/total

def f_score(true_labels, pred_labels):
    stats = calc_stats(true_labels, pred_labels)
    return stats['f1']

def acc(l1,l2):
    total = len(l1)
    correct = np.sum(l1==l2)
    return correct/total


# cross_validation(diabetes,10)

# stratified_cross_validation(diabetes, 2)



def print_plot(xaxis, data1, data2, bar1_name, bar2_name):
    pos = list(range(len(xaxis)))
    # pos = np.arange(len(xaxis))
    width=0.3
    fig, ax = plt.subplots(ncols=1, nrows=1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    ind = np.arange(len(xaxis))
    bar1 = ax.bar(pos, data1, width=0.3, align='center')
    bar2 = ax.bar([p + width for p in pos], data2, width=0.3,align='center')
    
    ax.set_xticklabels(xaxis, visible=True, rotation='vertical')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Number of folds')
    # ax.set_title(method)
    ax.set_xticks(ind + width )
    ax.legend( (bar1[0], bar2[0]), (bar1_name, bar2_name))
    for i, (v1, v2) in enumerate(zip(data1,data2)):
        ax.text(i-.10, v1 + .01, str(v1)+'%', color='black', ha='center')
        ax.text(i+.18, v2 + .01, str(v2)+'%', color='black',ha='center')
    plt.tight_layout()
    plt.ylim(0,+110)
    plt.show()


def separate_by_class(df):
    separated = {}
    unique_classes = np.unique(df[df.columns[-1]])
    for clas in unique_classes:
        d = df.loc[df[df.columns[-1]] == clas]
        separated[clas] = d
    return separated

def summarize(df):
    return [(df[colname].mean(),df[colname].std()) for colname in df.columns[:-1]]

def summarize_by_class(df):
    separated = separate_by_class(df)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    
    return summaries

def calculate_prob(x, mean, stdev):
    # print(x,stdev)
    if stdev == 0:
        return 0
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

def calculate_class_probs(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculate_prob(x, mean, stdev)

    return probabilities


def predict(summaries, inputVector):
	probabilities = calculate_class_probs(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
    # print('Summaries',summaries)
    predictions = []
    for index, row in testSet.iterrows():
        result = predict(summaries, list(row))
        predictions.append(result)
    return np.array(predictions)

msk = np.random.rand(len(diabetes)) < 0.8
train = diabetes[msk]
test = diabetes[~msk]
# print(f_score(test.iloc[:,-1], getPredictions(summarize_by_class(diabetes),test)))

def plot(xaxis,data,labels,ylabel,xlabel,title,ylim):
    ddict = {}
    for d,l in zip(data,labels):
        ddict[l]=d
    print(ddict)
    df = pd.DataFrame(ddict, index=xaxis)
    # df=df.astype(float)
    ax = df.plot.bar(rot=0)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.04, p.get_height() * 1.03),rotation = 45)
    plt.ylim(0,ylim)
    patches, labels = ax.get_legend_handles_labels()

    ax.legend(patches, labels, loc='lower right')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    # plt.legend((p1[0], p2[0]), ('Men', 'Women'))

    plt.show()

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,title,normalize=False,cmap=plt.cm.Blues):
   
    # if not title:
    #     if normalize:
    #         title = 'Normalized confusion matrix'
    #     else:
    #         title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def train():
    folds = [2,3,5,10,15]
    # f = 10
    bins = [3,5,9,17,33]
    b = 9
    data_cross = []
    data_cross_strat = []
    data_discret_len=[]
    data_discret_random=[]
    data_discret_entropy=[]
    data_normal = []
    data_bayes = []
    for f in folds:
        # data_discret_len.append(round(cross_validation(diabetes,f,b,'len','accuracy'),2))
        # data_discret_random.append(round(cross_validation(diabetes,f,b,'random','accuracy'),2))
        # data_discret_entropy.append(round(cross_validation(diabetes,f,b,'entropy','accuracy'),2))
        # data_cross.append(round(cross_validation(diabetes,f,b,'len','f'),2))
        # data_cross_strat.append(round(stratified_cross_validation(diabetes,f,b,'len','f'),2))
        # data_cross.append(round(cross_validation(wine,f,b,'len','accuracy'),2))
        # data_cross_strat.append(round(stratified_cross_validation(wine,f,b,'len','accuracy'),2))
        data_normal.append(round(cross_validation_normal(wine,f,b,'len','fmulti'),2))
        data_bayes.append(round(cross_validation(wine,f,b,'len','fmulti'),2))
    # print_plot(folds,data_cross,data_cross_strat,'CV','CV Stratified')

    # plot(folds,[data_cross,data_cross_strat],['CV','CV Stratified'],'Accuracy [%]','Number of folds', 'Crossvalidation vs Stratified Crossvalidation',100)

    # plot(folds,[data_cross,data_cross_strat],['CV','CV Stratified'],'F1 Score','Number of folds')
    plot(folds,[data_normal,data_bayes],['Normal Distribusion','Discretised'],'F1 Score','Number of folds','Rozkład normalny vs dyskretyzacja',1.1)

    # plot(bins,[data_discret_len,data_discret_random,data_discret_entropy],['equal','linear','entropy'],'Accuracy [%]','Number of categories','Porównanie metod dyskretyzacji',100)

train()
