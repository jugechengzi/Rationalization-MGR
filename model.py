import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.distributions.binomial as binomial

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        """
        Inputs:
        x -- (batch_size, seq_length)
        Outputs
        shape -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)







class Multi_gen(nn.Module):
    def __init__(self,args):
        super(Multi_gen, self).__init__()
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)


        self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        self.z_dim = 2
        self.dropout = nn.Dropout(args.dropout)
        self.gen_list=[]
        if args.share==0:
            for idx in range(args.num_gen):
                # temp=nn.Sequential(nn.GRU(input_size=args.embedding_dim,
                #               hidden_size=args.hidden_dim // 2,
                #               num_layers=args.num_layers,
                #               batch_first=True,
                #               bidirectional=True),
                #                            SelectItem(0),
                #                            nn.LayerNorm(args.hidden_dim),
                #                            self.dropout,
                #                            nn.Linear(args.hidden_dim, self.z_dim)).to('cuda:{}'.format(args.gpu))
                self.gen_list.append(nn.Sequential(nn.GRU(input_size=args.embedding_dim,
                              hidden_size=args.hidden_dim // 2,
                              num_layers=args.num_layers,
                              batch_first=True,
                              bidirectional=True),
                                           SelectItem(0),
                                           nn.LayerNorm(args.hidden_dim),
                                           self.dropout,
                                           nn.Linear(args.hidden_dim, self.z_dim)).to('cuda:{}'.format(args.gpu)))
        elif args.share==1:
            self.gen = nn.GRU(input_size=args.embedding_dim,
                              hidden_size=args.hidden_dim // 2,
                              num_layers=args.num_layers,
                              batch_first=True,
                              bidirectional=True)

            for idx in range(args.num_gen):
                self.gen_list.append(nn.Sequential(self.gen,
                                                   SelectItem(0),
                                                   nn.LayerNorm(args.hidden_dim),
                                                   self.dropout,
                                                   nn.Linear(args.hidden_dim, self.z_dim)).to('cuda:{}'.format(args.gpu)))


    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits=[gen(embedding) for gen in self.gen_list ]
        # gen_logits=self.generator(embedding)
        ########## Sample ##########
        # z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        z_list=[self.independent_straight_through_sampling(logit) for logit in gen_logits]
        ########## Classifier ##########
        cls_logits_list=[]
        for z in z_list:
            cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
            cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
            cls_outputs = cls_outputs * masks_ + (1. -
                                                  masks_) * (-1e6)
            # (batch_size, hidden_dim, seq_length)
            cls_outputs = torch.transpose(cls_outputs, 1, 2)
            cls_outputs, _ = torch.max(cls_outputs, axis=2)
            # shape -- (batch_size, num_classes)
            cls_logits = self.cls_fc(self.dropout(cls_outputs))
            cls_logits_list.append(cls_logits)
        return z_list, cls_logits_list

    def test(self,inputs, masks):
        masks_ = masks.unsqueeze(-1)

        ########## Genetator ##########
        embedding = masks_ * self.embedding_layer(inputs)  # (batch_size, seq_length, embedding_dim)
        # gen_output, _ = self.gen(embedding)  # (batch_size, seq_length, hidden_dim)
        # if self.lay:
        #     gen_output = self.layernorm1(gen_output)
        # gen_logits = self.gen_fc(self.dropout(gen_output))  # (batch_size, seq_length, 2)
        gen_logits = [torch.softmax(gen(embedding),dim=-1) for gen in self.gen_list]
        mean_logits=sum(gen_logits)/len(gen_logits)
        ########## Sample ##########
        # z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        # z_list = [self.independent_straight_through_sampling(logit) for logit in gen_logits]
        # z=self.independent_straight_through_sampling(mean_logits)
        z_distribution=binomial.Binomial(1,mean_logits)
        z=z_distribution.sample()
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits
    def test_one_head(self,inputs, masks):
        head_1=self.gen_list[0]
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)
        gen_logits=head_1(embedding)

        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)
        ########## Classifier ##########
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  # (batch_size, seq_length, embedding_dim)
        cls_outputs, _ = self.cls(cls_embedding)  # (batch_size, seq_length, hidden_dim)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        # shape -- (batch_size, num_classes)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def get_cls_param(self):
        layers = [self.cls, self.cls_fc]
        params = []
        for layer in layers:
            params.extend([param for param in layer.parameters() if param.requires_grad])
        return params





    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        # (batch_size, seq_length, embedding_dim)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  # (batch_size, seq_length, hidden_dim)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        # (batch_size, hidden_dim, seq_length)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        # shape -- (batch_size, num_classes)
        logits = self.cls_fc(self.dropout(outputs))
        return logits















