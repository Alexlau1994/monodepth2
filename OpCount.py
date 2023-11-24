import torch
from thop import profile
from fvcore.nn import FlopCountAnalysis

import networks


rn_encoder = networks.ResnetEncoder(num_layers=18, pretrained=False)
efn_encoder = networks.efficientnet_b0()
rn_decoder = networks.DepthDecoder(num_ch_enc=rn_encoder.num_ch_enc, scales=[0])
efn_decoder = networks.DepthDecoder(num_ch_enc=efn_encoder.num_ch_enc, scales=[0])

rn_encoder.eval()
efn_encoder.eval()

input1 = torch.randn(1, 3, 192, 640)
input2 = torch.randn(1, 3, 320, 1024)
rn_features1 = rn_encoder(input1)
rn_features2 = rn_encoder(input2)
efn_features1 = efn_encoder(input1)
efn_features2 = efn_encoder(input2)


macs, params = profile(rn_encoder, inputs=(input1, ), verbose=False)
print("ResnetEncoder Rosoultion: 640x192 MACs(G): {:.2f}, Params(M): {:.2f}".format(macs/1000000000, params/1000000))
macs, params = profile(rn_decoder, inputs=(rn_features1, ), verbose=False)
print("ResnetDecoder Rosoultion: 640x192 MACs(G): {:.2f}, Params(M): {:.2f}".format(macs/1000000000, params/1000000))
macs, params = profile(rn_encoder, inputs=(input2, ), verbose=False)
print("ResnetEncoder Rosoultion: 1024x320 MACs(G): {:.2f}, Params(M): {:.2f}".format(macs/1000000000, params/1000000))
macs, params = profile(rn_decoder, inputs=(rn_features2, ), verbose=False)
print("ResnetDecoder Rosoultion: 1024x320 MACs(G): {:.2f}, Params(M): {:.2f}".format(macs/1000000000, params/1000000))


macs, params = profile(efn_encoder, inputs=(input1, ), verbose=False)
print("EfficientnetEncoder Rosoultion: 640x192 MACs(G): {:.2f}, Params(M): {:.2f}".format(macs/1000000000, params/1000000))
macs, params = profile(efn_decoder, inputs=(efn_features1, ), verbose=False)
print("EfficientnetDecoder Rosoultion: 640x192 MACs(G): {:.2f}, Params(M): {:.2f}".format(macs/1000000000, params/1000000))
macs, params = profile(efn_encoder, inputs=(input2, ), verbose=False)
print("EfficientnetEncoder Rosoultion: 1024x320 MACs(G): {:.2f}, Params(M): {:.2f}".format(macs/1000000000, params/1000000))
macs, params = profile(efn_decoder, inputs=(efn_features2, ), verbose=False)
print("EfficientnetDecoder Rosoultion: 1024x320 MACs(G): {:.2f}, Params(M): {:.2f}".format(macs/1000000000, params/1000000))