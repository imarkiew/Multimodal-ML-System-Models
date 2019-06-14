from matplotlib import pyplot as plt
import pandas as pd


train_loss_no_aff = pd.read_csv('./noAff/epoch_loss.csv', delimiter=',')
val_loss_no_aff = pd.read_csv('./noAff/epoch_val_loss.csv', delimiter=',')
train_loss_no_aff['Step'] = train_loss_no_aff['Step'] + 1
val_loss_no_aff['Step'] = val_loss_no_aff['Step'] + 1
f = plt.figure()
plt.plot(train_loss_no_aff['Step'], train_loss_no_aff['Value'], 'r', val_loss_no_aff['Step'], val_loss_no_aff['Value'], 'b')
number_of_nodes = [i + 1 for i in range(len(train_loss_no_aff['Step']))]
plt.xticks(number_of_nodes)
plt.title('Crossentropy loss')
plt.xlabel('Epoka')
plt.ylabel('Wartość')
plt.legend(['trenowanie', 'walidacja'])
plt.grid()
f.savefig('./pdfs/losses.pdf', bbox_inches='tight')
plt.show()

train_categorical_accuracy_no_aff = pd.read_csv('./noAff/epoch_categorical_accuracy.csv', delimiter=',')
val_categorical_accuracy_no_aff = pd.read_csv('./noAff/epoch_val_categorical_accuracy.csv', delimiter=',')
train_categorical_accuracy_no_aff['Step'] = train_categorical_accuracy_no_aff['Step'] + 1
val_categorical_accuracy_no_aff['Step'] = val_categorical_accuracy_no_aff['Step'] + 1
f = plt.figure()
plt.plot(train_categorical_accuracy_no_aff['Step'], train_categorical_accuracy_no_aff['Value'], 'r', val_categorical_accuracy_no_aff['Step'], val_categorical_accuracy_no_aff['Value'], 'b')
number_of_nodes = [i + 1 for i in range(len(train_categorical_accuracy_no_aff['Step']))]
plt.xticks(number_of_nodes)
plt.title('Accuracy')
plt.xlabel('Epoka')
plt.ylabel('Wartość')
plt.legend(['trenowanie', 'walidacja'])
plt.grid()
f.savefig('./pdfs/accuracies.pdf', bbox_inches='tight')
plt.show()

lr_no_aff = pd.read_csv('./noAff/epoch_lr.csv', delimiter=',')
lr_no_aff['Step'] = lr_no_aff['Step'] + 1
f = plt.figure()
plt.plot(lr_no_aff['Step'], lr_no_aff['Value'], 'b')
number_of_nodes = [i + 1 for i in range(len(lr_no_aff['Step']))]
plt.xticks(number_of_nodes)
plt.title('Współczynnik szybkości uczenia')
plt.xlabel('Epoka')
plt.ylabel('Wartość')
plt.grid()
f.savefig('./pdfs/lrNoAff.pdf', bbox_inches='tight')
plt.show()

train_loss_aff = pd.read_csv('./aff/epoch_loss.csv', delimiter=',')
val_loss_aff = pd.read_csv('./aff/epoch_val_loss.csv', delimiter=',')
train_loss_aff['Step'] = train_loss_aff['Step'] + 1
val_loss_aff['Step'] = val_loss_aff['Step'] + 1
f = plt.figure()
plt.plot(train_loss_aff['Step'], train_loss_aff['Value'], 'r', val_loss_aff['Step'], val_loss_aff['Value'], 'b')
number_of_nodes = [i + 1 for i in range(len(train_loss_aff['Step']))]
plt.xticks(number_of_nodes)
plt.title('Crossentropy loss')
plt.xlabel('Epoka')
plt.ylabel('Wartość')
plt.legend(['trenowanie', 'walidacja'])
plt.grid()
f.savefig('./pdfs/lossesAff.pdf', bbox_inches='tight')
plt.show()

train_categorical_accuracy_aff = pd.read_csv('./aff/epoch_categorical_accuracy.csv', delimiter=',')
val_categorical_accuracy_aff = pd.read_csv('./aff/epoch_val_categorical_accuracy.csv', delimiter=',')
train_categorical_accuracy_aff['Step'] = train_categorical_accuracy_aff['Step'] + 1
val_categorical_accuracy_aff['Step'] = val_categorical_accuracy_aff['Step'] + 1
f = plt.figure()
plt.plot(train_categorical_accuracy_aff['Step'], train_categorical_accuracy_aff['Value'], 'r', val_categorical_accuracy_aff['Step'], val_categorical_accuracy_aff['Value'], 'b')
number_of_nodes = [i + 1 for i in range(len(train_categorical_accuracy_aff['Step']))]
plt.xticks(number_of_nodes)
plt.title('Accuracy')
plt.xlabel('Epoka')
plt.ylabel('Wartość')
plt.legend(['trenowanie', 'walidacja'])
plt.grid()
f.savefig('./pdfs/accuraciesAff.pdf', bbox_inches='tight')
plt.show()

lr_aff = pd.read_csv('./aff/epoch_lr.csv', delimiter=',')
lr_aff['Step'] = lr_aff['Step'] + 1
f = plt.figure()
plt.plot(lr_aff['Step'], lr_aff['Value'], 'b')
number_of_nodes = [i + 1 for i in range(len(lr_aff['Step']))]
plt.xticks(number_of_nodes)
plt.title('Współczynnik szybkości uczenia')
plt.xlabel('Epoka')
plt.ylabel('Wartość')
plt.grid()
f.savefig('./pdfs/lrAff.pdf', bbox_inches='tight')
plt.show()
