import torch


# Train the network
def train_net(
    net, train_iter, epochs, optimizer, device="cpu", scheduler=None, print_interval=10
):
    loss_list = []
    net = net.to(device)

    print("Train Start:")

    for e in range(epochs):
        # Certain layers behave differently depending on whether we're training
        # or testing, so tell the model to run in training mode
        net.train()
        state = None

        for train_data in train_iter:
            train_data = [ds.to(device) for ds in train_data]

            # Because of how the data is wrapped up into the DataLoader in `load_data`,
            # the first three things are pc, delta_in, and types, and the last thing
            # is the targets (delta_out).
            X = train_data[:-1]
            target = train_data[-1]

            loss, out, state = net(X, state, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss)

            # Detach state gradients to avoid autograd errors
            state = tuple([s.detach() for s in list(state)])

        if scheduler != None:
            scheduler.step()

        if (e + 1) % print_interval == 0:
            print(f"\tEpoch {e+1}\tLoss:\t{loss_list[-1]:.8f}")

    return loss_list


def prob_acc(pred, target):
    # pred shape: (N, K = 10)
    # target: (N, 1)
    num_correct = 0
    
    for expected_delta, predictions in zip(target, pred):
        if expected_delta in predictions:
            num_correct += 1

    return num_correct / len(target)


# Evaluate the network on a labeled dataset
def eval_net(net, eval_iter, device="cpu", state=None):
    train_acc = state == None
    net.eval()
    prob_acc_list = []

    for i, eval_data in enumerate(eval_iter):
        eval_data = [ds.to(device) for ds in eval_data]
        X = eval_data[:-1]
        target = eval_data[-1]

        preds, state = net.predict(X, state)

        acc = prob_acc(preds.cpu(), target.cpu())
        prob_acc_list.append(acc)

    if train_acc:
        print("Training Prob Acc.: {:.4f}".format(torch.tensor(prob_acc_list).mean()))
    else:
        print("Val Prob Acc.: {:.4f}".format(torch.tensor(prob_acc_list).mean()))

    return state