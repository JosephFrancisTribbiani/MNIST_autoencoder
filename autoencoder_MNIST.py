import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
  def __init__(self):
    super().__init__()  

    def init_normal(m):
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform(m.weight)  
        m.bias.data.fill_(0.)

    self.encoder = nn.Sequential(self.conv_block(in_channels=1, out_channels=16),
                                 self.conv_block(in_channels=16, out_channels=32, padding=0), 
                                 self.conv_block(in_channels=32, out_channels=64),
                                 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), 
                                 nn.Flatten())
    
    self.decoder = nn.Sequential(nn.Unflatten(1, (128, 1, 1)),
                                 nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3),
                                 self.deconv_block(in_channels=64, out_channels=32), 
                                 self.deconv_block(in_channels=32, out_channels=16, padding=0), 
                                 self.deconv_block(in_channels=16, out_channels=1))
    self.apply(init_normal)

  def forward(self, x):
    latent_code = self.encoder(x)
    reconstruction = self.decoder(latent_code)
    return reconstruction, latent_code

  def conv_block(self, in_channels, out_channels, k_conv=3, padding=1, k_pooling=2):
    block = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=k_conv,
                                    padding=padding), 
                          nn.BatchNorm2d(num_features=out_channels), 
                          nn.MaxPool2d(kernel_size=k_pooling),
                          nn.ELU())
    return block

  def deconv_block(self, in_channels, out_channels, k_deconv=3, padding=1, scale_factor=2):
    block = nn.Sequential(nn.Upsample(scale_factor=scale_factor), 
                          nn.ConvTranspose2d(in_channels=in_channels,
                                             out_channels=out_channels, 
                                             kernel_size=k_deconv,
                                             padding=padding), 
                          nn.BatchNorm2d(out_channels), 
                          nn.ELU())
    return block