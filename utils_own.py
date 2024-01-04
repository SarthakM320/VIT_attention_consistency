import torch

# rewrite all using torch.flip

def horizontal_flip(x):
    x1 = torch.zeros_like(x)
    for k in range(x.shape[-3]):
        x1[:,:,x.shape[-3] - 1 - k,:,:] = x[:,:,k,:,:]
        

    return x1

def vertical_flip(x):
    x1 = torch.zeros_like(x)
    
    for k in range(x.shape[-4]):
        x1[:,x.shape[-4] - 1 - k,:,:,:] = x[:,k,:,:,:]
        # x1[:,:,:,:,x.shape[-2] - 1 - k,:] = x[:,:,:,:,k,:]

    return x1

def horizontal_flip_target(x):
    
    x1 = torch.zeros_like(x)
    for k in range(x.shape[-1]):
        x1[:,:,:,:,x.shape[-1] - 1 - k] = x[:,:,:,:,k]
        

    return x1

def vertical_flip_target(x):
    x1 = torch.zeros_like(x)
    
    for k in range(x.shape[-2]):
        x1[:,:,:,x.shape[-2] - 1 - k,:] = x[:,:,:,k,:]

    return x1
