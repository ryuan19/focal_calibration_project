import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class IQLoss(nn.Module):
    def __init__(self, div="chi", gamma=0, alpha=0.5, lambda_gp=0, size_average=True):
        super(IQLoss, self).__init__()
        self.div = div
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_gp = lambda_gp
        self.size_average = size_average
    
    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        
        loss, loss_dict = self.iq_loss(input, target)
        return loss

# Full IQ-Learn objective with other divergences and options
def iq_loss(self, logits, targets):
  
    # keep track of logZ
    v0 = torch.logsumexp(logits, dim=1).mean()
    loss_dict = {}
    loss_dict['LogZ'] = v0.item()

    #  calculate 1st term for IQ loss
    #  -E_(ρ_expert)[r]
    reward = logits

    with torch.no_grad():
        # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
        if self.div == "hellinger":
            phi_grad = 1/(1+reward)**2
        elif self.div == "kl":
            # original dual form for kl divergence (sub optimal)
            phi_grad = torch.exp(-reward-1)
        elif self.div == "kl2":
            # biased dual form for kl divergence
            phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
        elif self.div == "kl_fix":
            # our proposed unbiased form for fixing kl divergence
            phi_grad = torch.exp(-reward)
        elif self.div == "js":
            # jensen–shannon
            phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
        else:
            phi_grad = 1
    # Gather log_prob on expert labels
    loss = -(phi_grad * reward).gather(1, target)
    loss_dict['softq_loss'] = loss.mean().item()

    # calculate 2nd term for IQ loss, we show different sampling strategies
    if  self.method == "value_expert":
        # calculate LogZ (works offline)
        value_loss = torch.logsumexp(logits, dim=1)
        loss += value_loss
        loss_dict['value_loss'] = value_loss.mean().item()

    elif self.method == "value":
        # calculate LogZ (works online, if collecting new images like JEM) 
        value_loss = torch.logsumexp(logits, dim=1) +  torch.logsumexp(expert_logits, dim=1)
        loss += value_loss
        loss_dict['value_loss'] = value_loss.mean().item()
    else:
        raise ValueError(f'This sampling method is not implemented: {args.method.type}')

    if self.grad_pen:
        # add a gradient penalty to loss (Wasserstein_1 metric)
        gp_loss = agent.critic_net.grad_pen(logits,
                                            self.lambda_gp)
        loss_dict['gp_loss'] = gp_loss.mean().item()
        loss += gp_loss

    if self.div == "chi":
        # Use χ2 divergence (works offline)
        chi2_loss = 1/(4 * self.alpha) * (reward**2)
        loss += chi2_loss
        loss_dict['chi2_loss'] = chi2_loss.mean().item()

    loss_dict['total_loss'] = loss.mean().item()
    
    if self.size_average: 
        loss = loss.mean()
    else:
        loss = loss.sum()
      
    return loss, loss_dict
