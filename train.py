from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data
X, y = load_iris(return_X_y=True)

# Preprocessing
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)

# Model Define
clf = SVC()

# Model Train
clf.fit(train_x, train_y)

# Model Evaluate
pred_y = clf.predict(test_x)
test_acc = accuracy_score(pred_y, test_y)
print(f"test accuracy: {test_acc:.4f}")
