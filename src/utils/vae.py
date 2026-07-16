import torch
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')


class VanillaVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = in_channels
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.hidden_dims = hidden_dims
        
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= self.out_channels,
                                      kernel_size= 3, padding= 1),
                            # nn.Tanh()   # Use Tanh to ensure output is in [-1, 1]
                            nn.Sigmoid()  # ensure output is in [0, 1]
                            )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    
    
    

# class ConditionalVAE(nn.Module):

#     def __init__(self,
#                  in_channels: int,
#                  num_classes: int,
#                  latent_dim: int,
#                  hidden_dims: List = None,
#                  img_size:int = 64,
#                  **kwargs) -> None:
#         super(ConditionalVAE, self).__init__()

#         self.latent_dim = latent_dim
#         self.img_size = img_size
#         self.in_channels = in_channels

#         self.embed_class = nn.Linear(num_classes, img_size * img_size)
#         self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

#         modules = []
#         if hidden_dims is None:
#             hidden_dims = [32, 64, 128, 256, 512]

#         in_channels += 1 # To account for the extra label channel
#         # Build Encoder
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels=h_dim,
#                               kernel_size= 3, stride= 2, padding  = 1),
#                     nn.BatchNorm2d(h_dim),
#                     nn.LeakyReLU())
#             )
#             in_channels = h_dim

#         self.encoder = nn.Sequential(*modules)
#         self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
#         self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


#         # Build Decoder
#         modules = []

#         self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * 4)

#         hidden_dims.reverse()

#         for i in range(len(hidden_dims) - 1):
#             modules.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(hidden_dims[i],
#                                        hidden_dims[i + 1],
#                                        kernel_size=3,
#                                        stride = 2,
#                                        padding=1,
#                                        output_padding=1),
#                     nn.BatchNorm2d(hidden_dims[i + 1]),
#                     nn.LeakyReLU())
#             )



#         self.decoder = nn.Sequential(*modules)

#         self.final_layer = nn.Sequential(
#                             nn.ConvTranspose2d(hidden_dims[-1],
#                                                hidden_dims[-1],
#                                                kernel_size=3,
#                                                stride=2,
#                                                padding=1,
#                                                output_padding=1),
#                             nn.BatchNorm2d(hidden_dims[-1]),
#                             nn.LeakyReLU(),
#                             nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels,
#                                       kernel_size= 3, padding= 1),
#                             # nn.Tanh()   # Use Tanh to ensure output is in [-1, 1]
#                             nn.Sigmoid()  # ensure output is in [0, 1]
#                             )


#     def encode(self, input: Tensor) -> List[Tensor]:
#         """
#         Encodes the input by passing through the encoder network
#         and returns the latent codes.
#         :param input: (Tensor) Input tensor to encoder [N x C x H x W]
#         :return: (Tensor) List of latent codes
#         """
#         result = self.encoder(input)
#         result = torch.flatten(result, start_dim=1)

#         # Split the result into mu and var components
#         # of the latent Gaussian distribution
#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)

#         return [mu, log_var]

#     def decode(self, z: Tensor) -> Tensor:
#         result = self.decoder_input(z)
#         result = result.view(-1, 512, 2, 2)
#         result = self.decoder(result)
#         result = self.final_layer(result)
#         return result

#     def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
#         """
#         Will a single z be enough ti compute the expectation
#         for the loss??
#         :param mu: (Tensor) Mean of the latent Gaussian
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian
#         :return:
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu

#     def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
#         y = kwargs['labels'].float()
#         embedded_class = self.embed_class(y)
#         embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
#         embedded_input = self.embed_data(input)

#         x = torch.cat([embedded_input, embedded_class], dim = 1)
#         mu, log_var = self.encode(x)

#         z = self.reparameterize(mu, log_var)

#         z = torch.cat([z, y], dim = 1)
#         return  [self.decode(z), input, mu, log_var]

#     def loss_function(self,
#                       *args,
#                       **kwargs) -> dict:
#         recons = args[0]
#         input = args[1]
#         mu = args[2]
#         log_var = args[3]

#         kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
#         recons_loss =F.mse_loss(recons, input)

#         kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

#         loss = recons_loss + kld_weight * kld_loss
#         return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

#     def sample(self,
#                num_samples:int,
#                current_device: int,
#                **kwargs) -> Tensor:
#         """
#         Samples from the latent space and return the corresponding
#         image space map.
#         :param num_samples: (Int) Number of samples
#         :param current_device: (Int) Device to run the model
#         :return: (Tensor)
#         """
#         y = kwargs['labels'].float()
#         z = torch.randn(num_samples, self.latent_dim)
        
#         z = z.to(current_device)

#         z = torch.cat([z, y], dim=1)
#         samples = self.decode(z)
#         return samples

#     def generate(self, x: Tensor, **kwargs) -> Tensor:
#         """
#         Given an input image x, returns the reconstructed image
#         :param x: (Tensor) [B x C x H x W]
#         :return: (Tensor) [B x C x H x W]
#         """

#         return self.forward(x, **kwargs)[0]




class ConditionalVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.in_channels = in_channels

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # ========== Encoder ==========
        encoder_in_channels = in_channels * 2  # input + condition
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(encoder_in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            encoder_in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # ========== Decoder ==========
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        self.condition_channels = in_channels

        hidden_dims.reverse()
        decoder_in_channels = hidden_dims[0] + self.condition_channels

        modules = []
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(decoder_in_channels, hidden_dims[1],
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[1]),
                nn.LeakyReLU())
        )
        for i in range(1, len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=self.in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )


    def encode(self, x: Tensor) -> List[Tensor]:
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor, condition: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)

        # Downsample cond_img to match [B, C, 2, 2]
        cond_down = F.adaptive_avg_pool2d(condition, (2, 2))

        # Concatenate latent features and condition
        result = torch.cat([result, cond_down], dim=1)

        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor, **kwargs) -> List[Tensor]:
        cond = kwargs['condition']
        x_cat = torch.cat([x, cond], dim=1)
        mu, log_var = self.encode(x_cat)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z, cond)
        return [out, x, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        cond_img = kwargs['condition'].to(current_device)
        
        if cond_img.size(0) == 1 and num_samples > 1:
            cond_img = cond_img.repeat(num_samples, 1, 1, 1)
        return self.decode(z, cond_img)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x, **kwargs)[0]
