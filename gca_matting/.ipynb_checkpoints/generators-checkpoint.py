import torch
import torch.nn as nn

from . import encoders, decoders


class Generator(nn.Module):
    def __init__(self, encoder, decoder):

        super(Generator, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        self.encoder = encoders.__dict__[encoder]()

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder]()

    def forward(self, image, trimap):
        inp = torch.cat((image, trimap), dim=1)
        embedding, mid_fea = self.encoder(inp)
        alpha, info_dict = self.decoder(embedding, mid_fea)

        return alpha, info_dict


def get_generator(encoder, decoder):
    generator = Generator(encoder=encoder, decoder=decoder)
    return generator

def remove_prefix_state_dict(state_dict, prefix="module"):
    """
    remove prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix)+1:]] = state_dict[key].float()
    return new_state_dict

if __name__ == "__main__":
    gca_model = get_generator(encoder='resnet_gca_encoder_29', decoder='res_gca_decoder_22')
    gca_ckpt = torch.load("checkpoints\gca-dist-all-data.pth")
    gca_model.load_state_dict(remove_prefix_state_dict(gca_ckpt['state_dict']), strict=True)
    gca_model.eval()
    input = torch.rand(1, 3, 512, 512)
    output = gca_model(input, input)
    print(output[0].shape)
    print(output[1])