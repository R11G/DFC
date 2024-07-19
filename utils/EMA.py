import torch  

class EMA:
  def __init__(self, mu):
    self.mu = mu
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.device = torch.device(d)
    self.average = torch.tensor(1.0).float().to(self.device)

  def apply(self, x):
    _x = x.abs().mean(0)
    self.average = self.mu * _x + (1-self.mu) * self.average