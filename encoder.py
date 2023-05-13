import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms

import utils
import simclr


class Encoder(nn.Module):
  def __init__(self, obs_shape):
    super().__init__()

    assert len(obs_shape) == 3
    self.repr_dim = 32 * 35 * 35

    self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU())

    self.apply(utils.weight_init)

  def forward(self, obs):
    obs = obs / 255.0 - 0.5
    h = self.convnet(obs)
    h = h.view(h.shape[0], -1)
    return h


class ResEncoder(nn.Module):
  def __init__(self, frame_stack, obs_shape, repr_dim=1024):
    super(ResEncoder, self).__init__()
    self.model = resnet18(pretrained=True)
    self.frame_stack = frame_stack
    self.repr_dim = repr_dim
    self.transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224)
    ])

    for param in self.model.parameters():
      param.requires_grad = False

    self.model.fc = nn.Identity()
    obs_shape = [32, ] + [obs_shape[x] for x in range(3)]
    out_dim = self.forward_conv(torch.randn(obs_shape)).shape[1]
    self.fc = nn.Linear(out_dim, self.repr_dim)
    self.ln = nn.LayerNorm(self.repr_dim)

    # Initialization
    nn.init.orthogonal_(self.fc.weight.data)
    self.fc.bias.data.fill_(0.0)

  @torch.no_grad()
  def forward_conv(self, obs, flatten=True):
    obs = obs / 255.0 - 0.5
    batch, in_channel, width, height = obs.shape
    in_channel = in_channel // self.frame_stack
    obs = obs.view(batch * self.frame_stack, in_channel, width, height)

    for name, module in self.model._modules.items():
      obs = module(obs)
      if name == 'layer2':
        break

    _, out_channel, width, height = obs.shape
    conv = obs.view(batch, self.frame_stack, out_channel, width, height)
    conv_current = conv[:, 1:, :, :, :]
    conv_prev = conv_current - conv[:, :self.frame_stack - 1, :, :, :].detach()
    conv = torch.cat([conv_current, conv_prev], axis=1)
    conv = conv.view(batch, 2 * (self.frame_stack - 1) * out_channel, width, height)
    if flatten:
      conv = conv.view(conv.size(0), -1)
    return conv

  def forward(self, obs):
    return self.ln(self.fc(self.forward_conv(obs)))


class SimclrEncoder(nn.Module):
  def __init__(self, ckpt_path, frame_stack, obs_shape, repr_dim=1024, stacks=['cur', 'diff']):
    """
    @param ckpt_path:
    @param frame_stack:
    @param obs_shape:
    @param repr_dim:
    @param stacks: ['',] type, str inside list can only be 'cur', 'diff', 'prev', 'ori',
                  determines how to preprocess image
    """
    super(SimclrEncoder, self).__init__()
    self.model, _ = simclr.get_resnet(*simclr.name_to_params(ckpt_path))
    self.model.load_state_dict(torch.load(ckpt_path)['resnet'])
    self.transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224)
    ])

    for param in self.model.parameters():
      param.requires_grad = False

    self.model.fc = nn.Identity()
    self.repr_dim = repr_dim
    self.frame_stack = frame_stack
    self.stacks = stacks

    # test conv-net output and generate full-connnected layer
    obs_shape = [obs_shape[x] for x in range(3)]
    out_dim = self.forward_conv(torch.randn([32, ] + obs_shape)).shape[1]
    self.fc = nn.Linear(out_dim, self.repr_dim)
    self.ln = nn.LayerNorm(self.repr_dim)

    # initialize fc layer
    nn.init.orthogonal_(self.fc.weight.data)
    self.fc.bias.data.fill_(0.0)

  @torch.no_grad()
  def forward_conv(self, obs, flatten=True):
    obs = obs / 255.0 - 0.5
    batch, in_channel, width, height = obs.shape
    in_channel = in_channel // self.frame_stack
    obs = obs.view(batch * self.frame_stack, in_channel, width, height)
    # apply conv-nets
    conv = self.model.net(obs)
    _, out_channel, width, height = conv.shape
    conv = conv.view(batch, self.frame_stack, out_channel, width, height)
    # stacks all the information
    temp = {'ori': conv, 'cur': conv[:,1:], 'prev': conv[:, :-1]}
    temp['diff'] = temp['cur'] - temp['prev']
    conv = torch.cat([temp[x] for x in self.stacks], axis=1)
    conv = conv.view(batch, conv.shape[1] * out_channel, width, height) # merge dim 2 and 3
    if flatten:
      conv = conv.view(conv.size(0), -1)
    return conv

  def forward(self, obs):
    return self.ln(self.fc(self.forward_conv(obs)))
