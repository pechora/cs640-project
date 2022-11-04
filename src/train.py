import torch
import tqdm

from model import Net
from data import load_data

batch_size = 4
learning_rate = 0.01
epochs = 100

def train(x, t):
    x = torch.tensor(x, dtype=torch.float).cuda()
    t = torch.tensor(t, dtype=torch.float).cuda()

    x = x[:4400]
    t = t[:4400]
    x_test = x[4400:]
    t_test = t[4400:]

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
        predicted = model(torch.tensor(x_test, dtype=torch.float32))
        acc = (predicted.reshape(-1).detach().numpy().round() == t_test).mean()

        print(f"Epoch {epoch} :: {acc}")

def main():
    x, t = load_data("../data/n1_spindle.mat")
    train(x, t)

if __name__ == "__main__":
    main()
