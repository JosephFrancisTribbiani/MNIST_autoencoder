import torch
import numpy as np


def deproc_img(data):
  """
  The function converts numpy array with float values to numpy array [0, 255]
  Input:
  - numpy array
  """
  data = np.clip(data, 0, 1)
  data *= 255
  data = np.clip(data, 0, 255).astype("uint8")
  return data


def visualize(data, img_w, img_h, ncols=10, margin=5, depth=3):
  nrows = len(data) // ncols + (len(data) % ncols != 0)
  h = img_h*nrows + margin*(nrows + 1)
  w = img_w*ncols + margin*(ncols + 1)
  z_array = np.zeros((depth, h, w), dtype='uint8')

  imgs_iterator = iter(data)
  for row in range(nrows):
    y = margin + row*(img_h + margin)
    for col in range(ncols):
      curr_array = next(imgs_iterator, None)
      if curr_array is not None:
        curr_array = deproc_img(curr_array)
        x = margin + col*(img_w + margin)
        z_array[:, y:y + img_h, x:x + img_w] = curr_array[:, :, :]
      else:
        break
  return z_array


def homotopy(model,
             x_1: torch.tensor,
             x_2: torch.tensor,
             n_embedings: int=10):
  """
  The function visualize a smooth trnsition between two reconstructed images.
  Input:
  - model - trained model which contains two modules: encoder and decoder;
  - x_1: torch.tensor - first input image. Values should be between 0 and 1. Size (1, 28, 28);
  - x_2: torch.tensor - second input image. Values should be between 0 and 1. Size (1, 28, 28);
  - n_embedings: int=10 - quantity of smoothes images between the x_1 and x_2, reconstructed from their embeddings.
  Output:
  - numpy array with shape (1, h, w). Values between 0 and 255. 1st row - reconstructed images, 2nd - their real analogs.
  """
  x = torch.stack([x_1, x_2])
  n_embedings += 1

  model.to('cpu')
  model.eval()
  with torch.no_grad():
    embedding_1, embedding_2 = model.encoder(x)
    embedding_values = list()
    for i in range(0, n_embedings + 1):
      embedding = embedding_1*((n_embedings - i)/n_embedings) + embedding_2*(i/n_embedings)
      embedding_values.append(embedding)
    embedding_values = torch.stack(embedding_values)
    reconstruct = model.decoder(embedding_values)

    real = torch.zeros_like(reconstruct)
    real[0], real[-1] = x_1, x_2
    reconstruct = torch.cat([reconstruct, real])
    return visualize(reconstruct.detach().numpy(), img_w=28, img_h=28, ncols=n_embedings + 1, depth=1)