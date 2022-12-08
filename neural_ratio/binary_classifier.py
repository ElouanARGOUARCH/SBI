import torch
from torch import nn
from tqdm import tqdm

class BinaryClassifier(nn.Module):
    def __init__(self, label_1_samples, label_0_samples, hidden_dims):
        super().__init__()

        self.label_1_samples = label_1_samples
        self.N = label_1_samples.shape[0]
        self.label_0_samples = label_0_samples
        self.M = label_0_samples.shape[0]
        assert label_0_samples.shape[-1]==label_1_samples.shape[-1], 'mismatch in samples dimensions'
        self.p = label_0_samples.shape[-1]
        self.target_samples = torch.cat([label_0_samples, label_1_samples], dim = 0)
        self.labels = torch.tensor([0]*self.M + [1]*self.N)

        network_dimensions = [self.p] + hidden_dims + [1]
        network = []
        for h0, h1 in zip(network_dimensions, network_dimensions[1:]):
            network.extend([nn.Linear(h0, h1), nn.SiLU(), ])
        network.pop()
        self.logit_r = nn.Sequential(*network)

        self.loss_values=[]


    def loss(self, label_1_batch, label_0_batch):
        log_sigmoid = torch.nn.LogSigmoid()
        return -torch.sum(log_sigmoid(self.logit_r(label_1_batch)))-torch.sum(log_sigmoid(-self.logit_r(label_0_batch)))

    def log_density_ratio(self,x):
        return self.logit_r(x).squeeze(-1) + torch.log(torch.tensor([self.M/self.N]))

    def train(self, epochs, batch_size = None):
        self.para_list = list(self.parameters())

        self.optimizer = torch.optim.Adam(self.para_list, lr=5e-4)
        if batch_size is None:
            batch_size = self.label_1_samples.shape[0]
        dataset = torch.utils.data.TensorDataset(self.target_samples, self.labels)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        pbar = tqdm(range(epochs))
        for t in pbar:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for i, batch in enumerate(dataloader):
                labels_batch = batch[1]
                label_1_batch = batch[0][labels_batch ==1].to(device)
                label_0_batch = batch[0][labels_batch ==0].to(device)
                self.optimizer.zero_grad()
                batch_loss = self.loss(label_1_batch,label_0_batch)
                batch_loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                iteration_loss = torch.tensor([self.loss(batch[0][batch[1] ==1].to(device),batch[0][batch[1] ==0].to(device)) for i, batch in enumerate(dataloader)]).mean().item()
            self.loss_values.append(iteration_loss)
            pbar.set_postfix_str('loss = ' + str(round(iteration_loss,6)))
        self.to(torch.device('cpu'))