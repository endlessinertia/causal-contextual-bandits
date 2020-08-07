import mnist_reader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

NUM_NEIGHS = 10
NUM_COMPS = 100

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

pca_model = PCA(NUM_COMPS)
X_train_pca = pca_model.fit_transform(X_train)
print(sum(pca_model.explained_variance_ratio_))

KNN = KNeighborsClassifier(NUM_NEIGHS)
# KNN.fit(X_train_pca, y_train)
scores = cross_val_score(KNN, X_train_pca, y_train, cv=6)
print(scores)

