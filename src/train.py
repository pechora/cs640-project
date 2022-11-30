import torch
import tqdm

from model import Net
from data import load_data, train_test_split

batch_size = 64
learning_rate = 0.01
epochs = 100
sample_rate = 128

def train(x, t):
    x = torch.tensor(x, dtype=torch.float).cuda()
    t = torch.tensor(t, dtype=torch.float).cuda()

    x, t, x_test, t_test = train_test_split(x, t, ratio=0.99)

    x_test = x_test.reshape((-1, 1, 38400))
    t_test = t_test.reshape(-1)

    model = Net(1, 1)

    model = model.float()
    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(epochs):
        model = model.train()

        for i in tqdm.tqdm(range(0, x.shape[0], batch_size)):
            x_batch = x[i: i + batch_size].reshape((-1, 1, 38400))
            t_batch = t[i: i + batch_size].reshape((-1, 1))

            output = model(x_batch)
            #print(output.shape, t_batch.shape)
            loss = loss_fn(output, t_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model = model.eval()
        predicted = model(x_test).reshape(-1)

        print(f"Prdicted: {predicted}\t Actual: {t_test}\t Prdicted Shape: {predicted.shape}\t Actual Shape: {t_test.shape}")
        acc = (t_test == predicted.round()).sum() / t_test.shape[0]

        print(f"Epoch {epoch} :: {acc}")
        torch.save(model.state_dict(), f"../model/model_{epoch}.pth")

def main():
    x, t = load_data("../data/n1_spindle.mat", sample_rate=sample_rate)
    train(x, t)

if __name__ == "__main__":
    main()