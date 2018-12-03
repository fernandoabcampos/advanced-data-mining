from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn import neighbors

nFolds = 4
metrics       = ['minkowski','euclidean','manhattan'] 
weights       = ['uniform','distance'] #10.0**np.arange(-5,4)
numNeighbors  = np.arange(5,10)
param_grid    = dict(metric = metrics, weights = weights, n_neighbors = numNeighbors)
cv            = cross_validation.StratifiedKFold(y, nFolds)
grid = GridSearchCV(neighbors.KNeighborsClassifier(),param_grid=param_grid,cv=cv)
grid.fit(dataImp,y)




from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC(gamma="scale")



# cv = For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
clf = GridSearchCV(svc, parameters, cv=4)
clf.fit(iris.data, iris.target)



i = 0

for wrong in wrong_list:
    
    if i < max_value:
        i += 1
        print('Error: {}, Correcto: {}'.format(wrong['Clasificación Equivocada'], wrong['Clasificación Correcta']))
        plt.imshow(images[wrong['Indice del error']].reshape(28, 28), cmap="gray")
        plt.set_title('Error: {}, Correcto: {}'.format(wrong['Clasificación Equivocada'], wrong['Clasificación Correcta']))
    else:
        break
        


        one_hot_encoded = pd.get_dummies(labels)





        import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.utils import np_utils

NB_CLASSES = 5

def create_model(learn_rate=0.01):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=100, activation='relu'))
    model.add(Dense(NB_CLASSES, activation='sigmoid'))
    # Compile model
    optimizer = SGD(lr=learn_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# One hot enconded
y_train_he = np_utils.to_categorical(y_train, NB_CLASSES)
y_test_he = np_utils.to_categorical(y_test, NB_CLASSES)

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1, 0.2]
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 20, 30, 40, 50]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, cv = 4, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train_pca, y_train_he)
