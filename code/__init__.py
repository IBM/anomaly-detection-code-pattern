import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
import numpy as np

"""[Paper Title : ] DAEMON: Unsupervised Anomaly Detection and Interpretation for Multivariate Time Series
"""

def weights_init(mod):
    """
    """
    classname = mod.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(mod.weight.data)
    elif classname.find("BatchNorm") != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform_(mod.weight)
        mod.bias.data.fill_(0.01)


"""
Encode the data
"""


class Encoder(nn.Module):
    """
    CNN Encoder
    """

    def __init__(self, input_dim, num_filter, latent_dim):
        """
        input_dim
        num_filter
        latent_dim
        """
        super(Encoder, self).__init__()

        self.softplus = nn.Softplus()
        self.main = nn.Sequential(

            nn.Conv1d(input_dim, num_filter, 4, 2, 1),
            nn.BatchNorm1d(num_filter),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(num_filter, num_filter * 2, 4, 2, 1),
            nn.BatchNorm1d(num_filter * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(num_filter * 2, num_filter * 4, 4, 2, 1),
            nn.BatchNorm1d(num_filter * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(num_filter * 4, num_filter * 8, 4, 2, 1),
            nn.BatchNorm1d(num_filter * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(num_filter * 8, num_filter * 16, 4, 2, 1),
            nn.BatchNorm1d(num_filter * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Conv1d(num_filter * 16, latent_dim, 4, 1, 0)
        self.log_var = nn.Conv1d(num_filter * 16, latent_dim, 4, 1, 0)

    def forward(self, input):
        """
        forward
        """
        
        output = self.main(input)
        mu = self.mu(output)
        log_var = self.softplus(self.log_var(output)) + 1e-4
        
        return mu, log_var


"""
Decode the data
"""


class Decoder(nn.Module):
    """
    CNN Decoder
    """

    def __init__(self, input_dim, num_filter, latent_dim):
        """
        Init args
        input_dim : 
        num_filter :
        latent_dim :
        """
        self.main = nn.Sequential(

            nn.ConvTranspose1d(latent_dim, num_filter * 16, 4, 1, 0),
            nn.BatchNorm1d(num_filter * 16),
            nn.ReLU(True),

            nn.ConvTranspose1d(num_filter * 16, num_filter * 8, 4, 2, 0),
            nn.BatchNorm1d(num_filter * 8),
            nn.ReLU(True),

            nn.ConvTranspose1d(num_filter * 8, num_filter * 4, 4, 2, 0),
            nn.BatchNorm1d(num_filter * 4),
            nn.ReLU(True),

            nn.ConvTranspose1d(num_filter * 4, num_filter * 2, 4, 2, 0),
            nn.BatchNorm1d(num_filter * 2),
            nn.ReLU(True),

            nn.ConvTranspose1d(num_filter * 2, num_filter, 4, 2, 1),
            nn.BatchNorm1d(num_filter),
            nn.ReLU(True),

            nn.ConvTranspose1d(num_filter, input_dim, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        """
        forward
        """
        output = self.main(input)
        return output


class Discriminator_Reconstruct(nn.Module):
    """
    Reconstruction based Discriminator
    """

    def __init__(self, input_dim, num_filter):
        """
        init
        input_dim, 
        num_filter
        """
        super(Discriminator_Reconstruct, self).__init__()
        model = Encoder(input_dim, num_filter, 1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:])
        self.classifier = nn.Sequential(
            nn.Conv1d(num_filter * 16, 1, 4, 1, 0), nn.Sigmoid()
        )

    def forward(self, x):
        """
        forward
        """
        features = self.features(x)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


class Discriminator_Latent(nn.Module):
    """[summary]

    Args:
        nn ([type]): [description]
    """

    def __init__(self, num_filter):
        super(Discriminator_Latent, self).__init__()

        self.features = nn.Sequential(

            nn.Conv1d(1, num_filter, 4, 2, 1),
            nn.BatchNorm1d(num_filter),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(num_filter, num_filter * 2, 4, 2, 1),
            nn.BatchNorm1d(num_filter * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(num_filter * 2, num_filter * 4, 4, 2, 1),
            nn.BatchNorm1d(num_filter * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(num_filter * 4, num_filter * 8, 4, 2, 1),
            nn.BatchNorm1d(num_filter * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(num_filter * 8, num_filter * 16, 4, 2, 1),
            nn.BatchNorm1d(num_filter * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(num_filter * 16, 1, 4, 1, 0), 
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


class Generator(nn.Module):
    """
    Generator
    """

    def __init__(self, input_dim, num_filter, latent_dim):
        """
        """
        super(Generator, self).__init__()
        self.encoder = Encoder(input_dim, num_filter, latent_dim)
        self.decoder = Decoder(input_dim, num_filter, latent_dim)

    def reparameter(self, mu, log_var):
        """
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        """
        mu, log_var = self.encoder(x)
        latent_z = self.reparameter(mu, log_var)
        output = self.decoder(latent_z)
        return output, latent_z, mu, log_var


class DAEMON:
    def __init__(
        self,
        input_dim,
        num_filter,
        latent_dim,
        device="cpu",
        niter=100,
        beta1=0.9,
        lr_d=0.0003,
        lr_g=0.001,
        w_rec=1,
        w_lat=1,
        train_batchsize=50,
    ):
        """
        """
        self.input_dim = input_dim
        self.num_filter = num_filter
        self.latent_dim = latent_dim
        self.device = device
        self.niter = niter
        self.beta1 = beta1
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.w_rec = w_rec
        self.w_lat = w_lat
        self.train_batchsize = train_batchsize

    def _pre_fit(self):
        """
        """
        self.G_ = Generator(self.input_dim, self.num_filter, self.latent_dim).to(
            self.device
        )
        self.G_.apply(weights_init)

        self.D_rec_ = Discriminator_Reconstruct(self.input_dim, self.num_filter).to(
            self.device
        )
        self.D_rec_.apply(weights_init)

        self.D_lat_ = Discriminator_Latent(self.num_filter).to(self.device)
        self.D_lat_.apply(weights_init)

        self.bce_criterion_ = nn.BCELoss()
        self.mse_criterion_ = nn.MSELoss()
        self.l1loss_ = nn.L1Loss()

        self.optimizer_D_rec_ = optim.Adam(
            self.D_rec_.parameters(), lr=self.lr_d, betas=(self.beta1, 0.999)
        )
        self.optimizer_D_lat_ = optim.Adam(
            self.D_lat_.parameters(), lr=self.lr_d, betas=(self.beta1, 0.999)
        )
        self.optimizer_G_ = optim.Adam(
            self.G_.parameters(), lr=self.lr_g, betas=(self.beta1, 0.999)
        )

    def fit(self, X, y=None):
        """
        """
        train_X = DataLoader(dataset=X, batch_size=self.train_batchsize, shuffle=True)
        self._pre_fit()
        start_time = time.time()
        for _ in range(self.niter):
            self.train_epoch_(train_X)
        total_train_time = time.time() - start_time
        print(total_train_time)
        return self

    def anomaly_score(self, X, **kwargs):
        """[summary]

        Args:
            X ([type]): [description]
        """
        predict_X = DataLoader(dataset=X, batch_size=1, shuffle=False)
        collector = []
        with torch.no_grad():
            for _, data in enumerate(predict_X, 0):
                input_data = data.permute([0,2,1]).float().to(self.device)
                fake, _, _, _ = self.G_(input_data)
                fake = fake.type(torch.DoubleTensor)
                data = data.type(torch.DoubleTensor)
                rec_error = torch.sum(torch.abs((fake.permute([0,2,1]) - data)), dim=2)
                collector.append(rec_error[:, -1])
        score = np.concatenate(collector, axis=0) 
        return score     

    def train_epoch_(self, X):
        """
        """
        # get the model ready
        self.G_.train()
        self.D_rec_.train()
        self.D_lat_.train()

        # bring X to DataLoader
        for data in X:
            train_batchsize = data.size(0)
            batch_x = data.permute([0, 2, 1]).float().to(self.device)
            p_z = torch.randn(train_batchsize, 1, self.latent_dim).to(self.device)

            self.update_d_rec_(train_batchsize, batch_x)
            self.update_d_lat_(p_z, batch_x)
            self.update_g_(p_z, batch_x)

    def update_d_rec_(self, train_batchsize, batch_x):
        """
        """
        real_label = 1.0
        fake_label = 0.0
        self.D_rec_.zero_grad()
        out_d_rec_real, _ = self.D_rec_(batch_x)
        out_g_fake, _, _, _ = self.G_(batch_x)
        out_d_rec_fake, _ = self.D_rec_(out_g_fake.detach())

        loss_d_rec_real = self.bce_criterion_(
            out_d_rec_real,
            torch.full((train_batchsize,), real_label, device=self.device),
        )
        loss_d_rec_fake = self.bce_criterion_(
            out_d_rec_fake,
            torch.full((train_batchsize,), fake_label, device=self.device),
        )

        self.loss_d_rec_ = loss_d_rec_real + loss_d_rec_fake
        self.loss_d_rec_.backward()
        self.optimizer_D_rec_.step()

    def update_d_lat_(self, train_batchsize, p_z, batch_x):
        real_label = 1.0
        fake_label = 0.0
        self.D_lat_.zero_grad()
        out_d_lat_real, _ = self.D_lat_(p_z)

        _, latent_z, _, _ = self.G_(batch_x)
        latent_z = latent_z.permute([0, 2, 1])
        out_d_lat_fake, _ = self.D_lat_(latent_z.detach())

        loss_d_lat_real = self.bce_criterion(
            out_d_lat_real,
            torch.full((train_batchsize,), real_label, device=self.device),
        )
        loss_d_lat_fake = self.bce_criterion(
            out_d_lat_fake,
            torch.full((train_batchsize,), fake_label, device=self.device),
        )

        loss_d_lat = loss_d_lat_real + loss_d_lat_fake
        loss_d_lat.backward()
        self.optimizer_D_lat_.step()

    def update_g_(self, p_z, batch_x):
        self.G_.zero_grad()
        
        out_g_fake, latent_z, _, _ = self.G_(batch_x)
        
        _, feat_rec_fake = self.D_rec_(out_g_fake)
        _, feat_rec_real = self.D_rec_(batch_x)

        latent_z = latent_z.permute([0, 2, 1])
        _, feat_lat_fake = self.D_lat_(latent_z)
        _, feat_lat_real = self.D_lat_(p_z)

        loss_g_rs = self.l1loss_(out_g_fake, batch_x)
        loss_g_rec = self.mse_criterion_(feat_rec_fake, feat_rec_real)
        loss_g_lat = self.mse_criterion_(feat_lat_fake, feat_lat_real)

        loss_g = loss_g_rs + (self.w_rec * loss_g_rec) + (self.w_lat * loss_g_lat)
        loss_g.backward()
        self.optimizer_G_.step()
