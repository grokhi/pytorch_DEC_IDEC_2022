import torch

from utils import cluster_acc

import time
from tqdm import tqdm

from autoencoders import StackedAutoEncoder
from deepclustering import DEC, IDEC


def train(
    model, loader, optimizer, loss_fn, device,
    static_loader=False
):
    model.train()
    model.to(device)

    if isinstance(model, DEC):
        if not model.initialized:
            xs = []
            ys = []
            
            if not static_loader:
                raise ValueError('Static loader was not received for further cluster centers intiialization ')

            for x, y in tqdm(static_loader, desc='Initialization of cluster centers'):
                x, y = x.to(device), y.to(device)
                x = x.reshape(-1, 28*28)
                xs.append(model.encoder(x).detach().cpu())
                ys.append(y)

            xs = torch.cat(xs)
            ys = torch.cat(ys)

            with torch.no_grad():
                model.kmeans.fit(xs)

                model.state_dict()["assignment.centroids"].copy_(
                    torch.tensor(model.kmeans.cluster_centers_)
                )

                model.init_acc = cluster_acc(ys.cpu().numpy(), model.kmeans.labels_)
                print('Initial cluster accuracy:', model.init_acc)
                
                time.sleep(1)

            model.y_pred_previous_initiialized = False # necessary for delta initialization 
                    
            model.initialized = True
            print('Centroids weights were initialized.')

    train_loss = 0
    rec_loss, cl_loss = 0, 0
    ys = []
    outputs = []
    counter = 0

    for x, y in tqdm(loader, desc='Train'):
        
        x, y = x.to(device), y.to(device)
        x = x.reshape(-1, 28*28)

        optimizer.zero_grad()

        if isinstance(model, DEC):
        
            if type(model) == DEC:
                q = model(x)
                p = model.get_target_distribution(q).detach()
                
                loss = loss_fn(q.log(), p)
                
            elif type(model) == IDEC:
                
                q, output = model(x)
                p = model.get_target_distribution(q).detach()

                loss = loss_fn(
                    q.log(), p,
                    output, x
                )

                rec_loss += loss_fn.rec_loss
                cl_loss += loss_fn.cl_loss

            ys.append(y)
            outputs.append(q.cpu().detach().argmax(1)) 

        elif isinstance(model, StackedAutoEncoder):
            output = model(x)
            loss = loss_fn(output, x)
        
        else:
            raise TypeError("Received model type is not supported. Use StackedAutoEncoder, StackedDenoisingAutoEncoder, DEC or IDEC")
        
        train_loss += loss.item()
        
    
        loss.backward()
        optimizer.step(closure=None)
        

    train_loss /= len(loader)


    if isinstance(model, DEC):
        
        y = torch.cat(ys).cpu().numpy()
        y_pred = torch.cat(outputs).numpy()
        
        if type(model) == DEC:
            return (
                train_loss, # float
                y, 
                y_pred,
            )
        if type(model) == IDEC:
            rec_loss /= len(loader)
            cl_loss /= len(loader)
            
            train_loss = {
                'total': train_loss,
                'rec' : rec_loss.detach().cpu().numpy(),
                'clust' : cl_loss.detach().cpu().numpy(),
            }
            return (
                train_loss, # dict
                y, 
                y_pred,
            )
    elif isinstance(model, StackedAutoEncoder):
        return train_loss


@torch.inference_mode()
def evaluate(
    model, loader, loss_fn, device
):
    model.eval()

    eval_loss = 0

    outputs = []
    ys = []

    for x, y in tqdm(loader, desc='Evaluation'):
        x, y = x.to(device), y.to(device)
        x = x.reshape(-1, 28*28)

        if isinstance(model, DEC):
            if type(model) == DEC:

                q = model(x)
                p = model.get_target_distribution(q).detach()
                
                loss = loss_fn(q.log(), p)
                
            elif type(model) == IDEC:
                q, output = model(x)
                p = model.get_target_distribution(q).detach()
                
                loss = loss_fn(
                    q.log(), p,
                    output, x
                )

            ys.append(y)
            outputs.append(q.cpu().detach().argmax(1))

        elif isinstance(model, StackedAutoEncoder):
            output = model(x)
            loss = loss_fn(output, x)

        else:
            raise TypeError("Received model type is not supported. Use StackedAutoEncoder, StackedDenoisingAutoEncoder, DEC or IDEC")

        eval_loss += loss.item()

    eval_loss /= len(loader)


    if isinstance(model, DEC):
        with torch.no_grad():
            y = torch.cat(ys).cpu().numpy()
            y_pred = torch.cat(outputs).numpy()
            
        return (
            eval_loss,
            y,
            y_pred
        )
    elif isinstance(model, StackedAutoEncoder):
        return eval_loss
        