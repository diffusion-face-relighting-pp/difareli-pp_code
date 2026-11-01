import torch

a = torch.load('/home/nontawat/diffae/checkpoints/ffhq256_autoenc/latent.pkl')
print(a['conds'].shape)
