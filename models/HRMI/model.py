import torch
from models.base import ModelBase
from utils.model_util import load_checkpoint, save_checkpoint
from torch import nn
import torch.nn.functional as F
from models.HRMI.encoder import *
import torch.nn.init as init


class HRMI(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, user_reputation_list, service_reputation_list,
                 user_country_one_hot, service_country_one_hot, g,
                 layers=None, EnablePositionalEncoding=False, num_encoder_layer=1, num_head=4, output_dim=1) -> None:
        super(HRMI, self).__init__()
        self.g = g

        self.reputation_dim = latent_dim // 8 * 2
        self.country_dim = latent_dim // 8 * 4
        self.id_dim = latent_dim // 8 * 2

        self.device = ("cuda" if (torch.cuda.is_available()) else "cpu")
        self.num_encoder_layer = num_encoder_layer

        self.layerNorm = LayerNorm(self.reputation_dim)  # for reputation

        self.user_country_one_hot = user_country_one_hot
        self.user_reputation_list = user_reputation_list
        self.user_reputation_linear = nn.Linear(1, self.reputation_dim)
        self.user_country_linear = nn.Linear(len(user_country_one_hot[0]), self.country_dim)

        self.service_country_one_hot = service_country_one_hot
        self.service_reputation_list = service_reputation_list
        self.service_reputation_linear = nn.Linear(1, self.reputation_dim)
        self.service_country_linear = nn.Linear(len(service_country_one_hot[0]), self.country_dim)

        self.MLP_embedding_user = nn.Embedding(num_embeddings=num_users,
                                               embedding_dim=self.id_dim)
        self.MLP_embedding_item = nn.Embedding(num_embeddings=num_items,
                                               embedding_dim=self.id_dim)

        self.MLP_layers = nn.ModuleList()
        self.MLP_layers.append(nn.Linear(latent_dim * 2, layers[0]))
        for in_size, out_size in zip(layers, layers[1:]):
            self.MLP_layers.append(nn.Linear(in_size, out_size))
        self.MLP_output = nn.Linear(layers[-1], latent_dim)

        self.linear_0_5 = nn.Linear(2 * latent_dim, latent_dim)
        self.linear_0_25 = nn.Linear(latent_dim, latent_dim // 2)

        self.linear_2 = nn.Linear(latent_dim // 2, latent_dim)
        self.linear_4 = nn.Linear(latent_dim, 2 * latent_dim)

        self.attn = MultiHeadAttention(num_head, 2 * latent_dim)
        self.ffn = PositionWiseFeedForward(2 * latent_dim, latent_dim)
        self.encoder_layer = EncoderLayer(2 * latent_dim, self.attn, self.ffn)
        self.encoder = Encoder(2 * latent_dim, 2 * latent_dim, 1, self.encoder_layer, EnablePositionalEncoding)

        self.attn_0_5 = MultiHeadAttention(num_head, latent_dim)
        self.ffn_0_5 = PositionWiseFeedForward(latent_dim, latent_dim)
        self.encoder_layer_0_5 = EncoderLayer(latent_dim, self.attn_0_5, self.ffn_0_5)
        self.encoder_0_5 = Encoder(latent_dim, latent_dim, 1, self.encoder_layer_0_5, EnablePositionalEncoding)

        self.attn_0_25 = MultiHeadAttention(num_head, latent_dim // 2)
        self.ffn_0_25 = PositionWiseFeedForward(latent_dim // 2, latent_dim)
        self.encoder_layer_0_25 = EncoderLayer(latent_dim // 2, self.attn_0_25, self.ffn_0_25)
        self.encoder_0_25 = Encoder(latent_dim // 2, latent_dim // 2, 1, self.encoder_layer_0_25,
                                    EnablePositionalEncoding)

        self.encoder_linear = nn.Linear(13 * latent_dim // 2, 2 * latent_dim)

        self.linear1 = nn.Linear(2 * latent_dim, latent_dim)
        self.linear2 = nn.Linear(latent_dim, latent_dim // 2)
        self.linear3 = nn.Linear(latent_dim // 2, output_dim)

    def forward(self, user_indexes, item_indexes):
        user_reputation, user_country = [], []
        for i in user_indexes:
            user_reputation.append([self.user_reputation_list[i]])
            user_country.append(self.user_country_one_hot[i])

        service_reputation, service_country = [], []
        for i in item_indexes:
            service_reputation.append([self.service_reputation_list[i]])
            service_country.append(self.service_country_one_hot[i])
        fac, ifac = (self.g - 1) / self.g, 1 / self.g
        for i in range(len(user_reputation)):
            user_reputation[i], service_reputation[i] = [x * fac + y * ifac for x, y in
                                                         zip(user_reputation[i], service_reputation[i])], [
                x * fac + y * ifac for x, y in zip(service_reputation[i], user_reputation[i])]

        user_reputation = torch.Tensor(user_reputation).to(self.device)
        u_r = user_reputation
        user_country = torch.Tensor(user_country).to(self.device)
        user_reputation_embedding = self.user_reputation_linear(user_reputation)
        user_country_embedding = self.user_country_linear(user_country)
        user_reputation_embedding = self.layerNorm(user_reputation_embedding)

        service_reputation = torch.Tensor(service_reputation).to(self.device)
        s_r = service_reputation
        service_country = torch.Tensor(service_country).to(self.device)
        service_reputation_embedding = self.service_reputation_linear(service_reputation)
        service_country_embedding = self.service_country_linear(service_country)
        service_reputation_embedding = self.layerNorm(service_reputation_embedding)

        ATTENTION_user_embedding = torch.cat(
            [self.MLP_embedding_user(user_indexes), user_reputation_embedding, user_country_embedding], dim=-1)
        ATTENTION_item_embedding = torch.cat(
            [self.MLP_embedding_item(item_indexes), service_reputation_embedding, service_country_embedding], dim=-1)

        u_vec = ATTENTION_user_embedding
        s_vec = ATTENTION_item_embedding

        x = torch.cat([ATTENTION_user_embedding, ATTENTION_item_embedding], dim=-1)

        for i in range(self.num_encoder_layer):
            x1 = self.encoder(x)
            a = torch.relu(self.linear_0_5(x))

            x2 = self.encoder_0_5(a)
            a = torch.relu(self.linear_0_25(a))

            x3 = self.encoder_0_25(a)
            a = torch.relu(self.linear_2(a))

            x4 = self.encoder_0_5(a)
            a = torch.relu(self.linear_4(a))

            x5 = self.encoder(a)

            x = torch.cat([x1, x2, x3, x4, x5], dim=-1)
            x = self.encoder_linear(x)

        ATTENTION_vec = x
        ATTENTION_vec = torch.relu(self.linear1(ATTENTION_vec))
        ATTENTION_vec = torch.relu(self.linear2(ATTENTION_vec))
        output = self.linear3(ATTENTION_vec)

        return output, u_r, s_r, u_vec, s_vec


class HRMIModel(ModelBase):
    def __init__(self, loss_fn, num_users, num_items, latent_dim, user_reputation, service_reputation,
                 user_country_one_hot, service_country_one_hot, g,
                 layers=None, output_dim=1, use_gpu=True, num_encoder_layer=1) -> None:
        super().__init__(loss_fn, use_gpu)
        self.name = __class__.__name__

        if layers is None:
            layers = [32, 8]
        self.model = HRMI(num_users, num_items, latent_dim, user_reputation, service_reputation,
                          user_country_one_hot, service_country_one_hot, g,
                          layers=layers, output_dim=output_dim, EnablePositionalEncoding=False,
                          num_encoder_layer=num_encoder_layer)


        if use_gpu:
            self.model.to(self.device)

    def parameters(self):
        return self.model.parameters()

    def __repr__(self) -> str:
        return str(self.model)
