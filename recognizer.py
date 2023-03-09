import os

import master_image
from sklearn.model_selection import train_test_split

import pickle
import torch
import torch.nn as nn

IMAGE_SIZE =50
CATEGORIES = []
filename = 'finalized_model.sav'
path = "7500-unique-kana-images/datasets"

class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 180)
        self.act = nn.ReLU()
        self.output = nn.Linear(180, 71)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x



if os.path.isfile(filename):
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))
    CATEGORIES = master_image.MasterImageThree(PATH=path, IMAGE_SIZE=IMAGE_SIZE).get_categories()
else:
    a = master_image.MasterImageThree(PATH=path, IMAGE_SIZE=IMAGE_SIZE)
    X_Dataset, y_Datset = a.load_dataset()
    CATEGORIES = a.get_categories()
    # split
    X_train, X_test, y_train, y_test = train_test_split(X_Dataset, y_Datset, train_size=0.7, shuffle=True)

    model = Multiclass()
    import numpy
    import torch.optim as optim
    import tqdm

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    numpy_things = numpy.unique(y_test)

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 200
    batch_size = 5
    batches_per_epoch = len(X) // batch_size


    for epoch in range(n_epochs):
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
                X_batch_train = X[start:start+batch_size]
                # print(X_batch)
                y_batch_train = y[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch_train)
                print(len(y_batch_train))
                print(len(y_batch_train[0]))
                print(len(y_batch_train[0][0]))
                #print(len(y_batch_train[0][0][0]))
                #print(y_batch_train[0][0][0])
                #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                #print (y_pred[0])
                #break
                loss = loss_fn(y_pred, y_batch_train)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()

    y_pred = model(X_test)
    ce = loss_fn(y_pred, y_test)
    acc = (torch.argmax(y_pred, 1) == torch.argmax(y_test, 1)).float().mean()
    print(f"Epoch {epoch} validation: Cross-entropy={float(ce)}, Accuracy={float(acc)}")
    # save the model to disk
    pickle.dump(model, open(filename, 'wb'))