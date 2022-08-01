import torch
import torch.nn as nn

import copy

class LinearEncoder(nn.Module):
    def __init__(self, shape, output_size):
        super(LinearEncoder, self).__init__()
        self.network = nn.Sequential(nn.Linear(shape.numel(), output_size*4),
                                     nn.ReLU(),
                                     nn.Linear(output_size*4, output_size*2),
                                     nn.ReLU(),
                                     nn.Linear(output_size*2, output_size),
                                     nn.ReLU())
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.network(x)

class ConvEncoder(nn.Module):
    def __init__(self, shape, output_size):
        super(ConvEncoder, self).__init__()

        conv_feats = [1, 64, 64, 64, 64]
        layers = []
        for i in range(len(conv_feats) -1):
            layers.append(nn.Conv2d(conv_feats[i], conv_feats[i+1], 3))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(3, stride=2))
        self.conv_net = nn.Sequential(*layers)
        num_elements = self.conv_net(torch.zeros(1, 1, *shape))
        self.linear = LinearEncoder(num_elements.shape[1:], output_size)
    
    def forward(self, x):
        feats = self.conv_net(x)
        return self.linear(feats)

class NetworkEncoderModel(nn.Module):
    def __init__(self,
                 example_network,
                 hidden_size=512,
                 output_size=512,
                 linear_feats=32,
                 conv_feats=128):
        super(NetworkEncoderModel, self).__init__()
        parameter_shapes = [p.shape for p in example_network.parameters()]
        
        self.networks = []
        total_feats = 0
        for shape in parameter_shapes:
            if len(shape) == 1 or (len(shape) == 2 and (shape[0] == 1 or shape[1] == 1)):
                self.networks.append(LinearEncoder(shape, linear_feats))
                total_feats += linear_feats
            else:
                self.networks.append(ConvEncoder(shape, conv_feats))
                total_feats += conv_feats


        self.linear_layers = nn.Sequential(nn.Linear(total_feats, hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(hidden_size, hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(hidden_size, output_size))

    def forward(self, x):
        tensors = []

        for model in x:
            for i, params in enumerate(model.parameters()):
                if i > len(tensors) - 1:
                    tensors.append([])
                tensors[i].append(params.detach())

        encodings = []
        for i in range(len(tensors)):
            tensors[i] = torch.stack(tensors[i])
            if tensors[i].shape[1] == 1 and len(tensors[i].shape) == 3:
                tensors[i] = tensors[i].reshape(tensors[i].shape[0], -1)
            elif len(tensors[i].shape) == 3:
                tensors[i] = tensors[i].unsqueeze(1)
            encodings.append(self.networks[i](tensors[i]))
      
        all_encodings = torch.cat(encodings, dim=-1)

        out = self.linear_layers(all_encodings)

        return out


if __name__ == '__main__':
    from isdf.modules import fc_map, embedding
    positional_encoding = embedding.PostionalEncoding(
        min_deg=0,
        max_deg=8,
        scale=0.04,
        transform = torch.eye(4),
    )

    network = fc_map.SDFMap(positional_encoding,
                            hidden_size=256,
                            hidden_layers_block=1,
                            scale_output=1.)
    
    encoder = NetworkEncoderModel(network)

    networks = [copy.deepcopy(network) for i in range(4)]

    out = encoder(networks)

    print("out", out.shape)

    print("numel", sum([p.numel() for p in encoder.parameters()]))