import numpy as np
import matplotlib.pyplot as plt
from rlr import rlr
import joblib

data = np.load("./data/data.npz")
Xtr = data["Xtr"] #Train inputs
Ytr = data["Ytr"] #Train labels
Ftr = data["Ftr"] #Train data file names

Xte = data["Xte"] #Test inputs
Yte = data["Yte"] #Test labels
Fte = data["Fte"] #Test data file names


# HW2 Part 1 Testing Functions
# X_test = np.array([[0.5,0.7],[-0.25,0.3]])
# Y_test = np.array([1,-1]).reshape(2,1)
# theta_test = np.array([-0.3,2,1.5]).T



#Set up example model and parameters
model = rlr()
N,D   = Xtr.shape
theta = np.zeros((D+1,0))
lam   = np.linspace(0,10,11)


#HW2 Part 1 Testing
# print(model.risk_grad(theta_test,X_test,Y_test))
# print(model.regularizer_grad(theta_test))
# print(model.regularized_risk_grad(theta_test,X_test,Y_test,0.1))
# print(model.risk(theta_test,X_test,Y_test))
# print(model.regularized_risk(theta_test,X_test,Y_test,0.1))
# print(model.regularizer(theta_test))
# print(model.fit(X_test,Y_test,0.1))

# lam1 = 5.7
# mod = model.fit(Xtr,Ytr,lam1)
# mod_pred = model.predict(mod,Xte)
# mod_pred = mod_pred * Yte
# mod_error = np.sum(mod_pred<0)/Xte.shape[0]
# print("Model Error:",mod_error)
# joblib.dump(mod, 'cough')

#Example call to fit:
out = []
for i in range(0,lam.shape[0]):
    out.append(model.fit(Xtr,Ytr,i))
training_error = []
testing_error = []
for i in range(0,lam.shape[0]):
    training_pred = model.predict(out[i],Xtr)
    testing_pred = model.predict(out[i],Xte)
    training_pred = training_pred * Ytr
    testing_pred = testing_pred * Yte
    training_error.append(np.sum(training_pred<0)/Xtr.shape[0])
    testing_error.append(np.sum(testing_pred<0)/Xte.shape[0])
# Errors vs Lambda
plt.figure()
plt.plot(lam,training_error,label="Training Error",alpha=1, linewidth=0.5, color='black', zorder=1)
plt.plot(lam,testing_error,label="Testing Error", alpha=0.4, linewidth=2.25, color='green', zorder=2)
plt.xlabel("Lambda")
plt.ylabel("Error")
plt.title("Errors vs Lambda")
plt.legend()
plt.show()

print("weights:",out)
print("Training Error:",training_error)
print("Testing Error:",testing_error)

weights_len = np.arange(0,out[0].shape[0])
plt.stem(weights_len, out[0],label="Lambda = 0", basefmt=' ', linefmt='r-', markerfmt='ro')
plt.stem(weights_len, out[5],label="Lambda = 5", basefmt=' ', linefmt='g-', markerfmt='go')
plt.stem(weights_len, out[10],label="Lambda = 10", basefmt=' ', linefmt='b-', markerfmt='bo')
plt.xlabel("Weight number")
plt.ylabel("Weights value")
plt.title("Weights values for different lambdas")
plt.legend()
plt.show()

#6k
abs_out = []
abs_out_percentage = []
for i in range(0,lam.shape[0]):
    abs_out.append(np.abs(out[i]))
    abs_out_percentage.append(np.sum(abs_out[i]<0.01)/out[i].shape[0])

plt.figure()
plt.plot(lam,abs_out_percentage)
plt.xlabel("Lambda")
plt.ylabel("Percentage of weights < 0.01")
plt.title("Percentage of weights < 0.01 vs Lambda")
plt.show()

#6l
indices_to_remove = []
for i in range(0,len(out)):
    indices_to_remove.append(np.where(np.abs(out[i][:-1])<0.01)[0])


out_sparse = []
#Creating Synthetic sparified dataset 
Xtr_sparse = []
Xte_sparse = []

for i in range(0,len(indices_to_remove)):
    out_sparse.append(np.delete(out[i],indices_to_remove[i],axis=0))
    Xtr_sparse.append(np.delete(Xtr,indices_to_remove[i],axis=1))
    Xte_sparse.append(np.delete(Xte,indices_to_remove[i],axis=1))
    print(Xtr_sparse[i].shape)
    print(Xte_sparse[i].shape)
    print(out_sparse[i].shape)

training_error_sparse = []
testing_error_sparse = []
for i in range(0,lam.shape[0]):
    training_pred = model.predict(out_sparse[i],Xtr_sparse[i])
    testing_pred = model.predict(out_sparse[i],Xte_sparse[i])
    training_pred = training_pred * Ytr
    testing_pred = testing_pred * Yte
    training_error_sparse.append(np.sum(training_pred<0)/Xtr_sparse[i].shape[0])
    testing_error_sparse.append(np.sum(testing_pred<0)/Xte_sparse[i].shape[0])

print("Training Error Sparse:",training_error_sparse)
print("Testing Error Sparse:",testing_error_sparse)

# Sparse Errors vs Lambda
plt.figure()
plt.plot(lam,training_error_sparse,label="Training Error",alpha=1, linewidth=0.5, color='black', zorder=1)
plt.plot(lam,testing_error_sparse,label="Testing Error", alpha=0.4, linewidth=2.25, color='green', zorder=2)
plt.xlabel("Lambda")
plt.ylabel("Error")
plt.title("Sparse Errors vs Lambda")
plt.legend()
plt.show()