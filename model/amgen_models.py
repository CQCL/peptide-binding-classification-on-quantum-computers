import torch
from torch import nn
import torchquantum as tq


class QRNN(tq.QuantumModule):
    def __init__(self,
                 encoder_ansatz,
                 encoder_layers=1,
                 reupload_count=1,
                 recurrent_ansatz=None,
                 recurrent_layers=1,
                 n_wires=4,
                 measure='all',
                 dropout_prob=0.0,
                 final='linear'):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        encoder_dict, self.num_params = encoder_ansatz(n_wires=n_wires, layers=encoder_layers)
        self.encoder = tq.GeneralEncoder(encoder_dict)
        self.reupload_count = reupload_count
        self.dropout = nn.Dropout(p=dropout_prob)

        if reupload_count > 1:
            self.reupload_weights = nn.Parameter(torch.randn(reupload_count, self.num_params))
            self.reupload_bias = nn.Parameter(torch.zeros(reupload_count, self.num_params))
        else:
            self.reupload_weights = nn.Parameter(torch.ones(1, self.num_params), requires_grad=False)
            self.reupload_bias = nn.Parameter(torch.zeros(1, self.num_params), requires_grad=False)

        if recurrent_ansatz is None:
            self.recurrent_circuit = tq.GeneralEncoder({})
            self.recurrent_params = None
        else:
            recurrent_dict, num_recurrent_params = recurrent_ansatz(n_wires=n_wires, layers=recurrent_layers)
            self.recurrent_circuit = tq.GeneralEncoder(recurrent_dict)
            self.recurrent_params = nn.Parameter(torch.rand(num_recurrent_params))

        if measure == 'all':
            self.measure = tq.measurement.MeasureOne(tq.PauliZ)
            self.final = nn.Linear(in_features=self.n_wires, out_features=1)
        elif measure == 'single':
            self.measure = tq.measurement.MeasureOne(tq.PauliZ)
            if final =="linear":
                self.final = nn.Linear(in_features=1, out_features=1)
            elif final == "rescale":
                def rescale(x):
                    return (x+1.)/2.
                self.final = rescale


class PureQRNN(QRNN):
    def __init__(self,
                 vocab_size,
                 encoder_ansatz,
                 encoder_layers=1,
                 reupload_count=1,
                 recurrent_ansatz=None,
                 recurrent_layers=1,
                 n_wires=4,
                 measure='all',
                 dropout_prob=0.0,
                 final='linear'):

        super().__init__(encoder_ansatz, encoder_layers, reupload_count,
                         recurrent_ansatz, recurrent_layers, n_wires, measure,
                         dropout_prob, final)

        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim=self.num_params,
                                      padding_idx=0)
        nn.init.uniform_(self.embedding.weight, a=0.0, b=1)
        self.embedding._fill_padding_idx_with_zero()

    def get_embeddings(self, x):
        lengths = (x == 0).long().argmax(1, keepdim=True) - 1
        lengths[lengths == -1] = x.shape[1] - 1
        return self.embedding(x), lengths

    def forward(self, x):
        self.q_device.reset_states(x.shape[0])

        x_embs, lengths = self.get_embeddings(x)
        x_embs = self.dropout(x_embs)

        measures = []
        for i in range(x.shape[1]):
            for j in range(self.reupload_count):
                self.encoder(self.q_device,
                             self.reupload_weights[j] * x_embs[:, i] + self.reupload_bias[j],
                             reset=False)
            if i != x.shape[1] - 1:
                self.recurrent_circuit(self.q_device, self.recurrent_params, reset=False)
            measures.append(self.measure(self.q_device))

        measures = torch.stack(measures, dim=1)
        lengths = lengths.unsqueeze(2).repeat(1, 1, measures.shape[2])
        out = measures.gather(1, lengths) #.squeeze()
        return self.final(out)


class SingleEmbedQRNN(QRNN):
    def __init__(self,
                 classical_embed_dim,
                 encoder_ansatz,
                 encoder_layers=1,
                 reupload_count=1,
                 recurrent_ansatz=None,
                 recurrent_layers=1,
                 n_wires=4,
                 measure='all',
                 dropout_prob=0.0,
                 final='linear'):
        super().__init__(encoder_ansatz, encoder_layers, reupload_count,
                         recurrent_ansatz, recurrent_layers, n_wires, measure,
                         dropout_prob, final)

        self.embed_transform = nn.Linear(classical_embed_dim, self.num_params, bias=False)
        self.end_token_params = nn.Parameter(torch.rand(self.num_params))

    def forward(self, x):
        self.q_device.reset_states(x.shape[0])

        x_embs = self.embed_transform(x)
        x_embs = torch.sigmoid(x_embs)
        x_embs = self.dropout(x_embs)
        for i in range(x.shape[1]):
            for j in range(self.reupload_count):
                self.encoder(self.q_device,
                             self.reupload_weights[j] * x_embs[:, i] + self.reupload_bias[j],
                             reset=False)

            self.recurrent_circuit(self.q_device, self.recurrent_params, reset=False)

        self.encoder(self.q_device, self.end_token_params.repeat(x.shape[0], 1), reset=False)

        out = self.measure(self.q_device)
        out = self.final(out)

        return out


class RandomEmbedQRNN(QRNN):
    def __init__(self,
                 vocab_size,
                 encoder_ansatz,
                 embed_dim=0,
                 encoder_layers=1,
                 reupload_count=1,
                 recurrent_ansatz=None,
                 recurrent_layers=1,
                 n_wires=4,
                 measure='all',
                 dropout_prob=0.0,
                 final='linear'):
        super().__init__(encoder_ansatz, encoder_layers, reupload_count,
                         recurrent_ansatz, recurrent_layers, n_wires, measure,
                         dropout_prob, final)

        if embed_dim == 0:
            embed_dim = self.num_params

        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim=embed_dim,
                                      padding_idx=0)
        nn.init.uniform_(self.embedding.weight, a=0.0, b=1)
        self.embedding._fill_padding_idx_with_zero()

        self.embed_transform = nn.Linear(embed_dim, self.num_params, bias=False)
        self.end_token_params = nn.Parameter(torch.rand(self.num_params))

    def get_embeddings(self, x):
        lengths = (x == 0).long().argmax(1, keepdim=True) - 1
        lengths[lengths == -1] = x.shape[1] - 1
        x_embs = self.embedding(x)
        x_embs = self.embed_transform(x_embs)
        x_embs = torch.sigmoid(x_embs)
        return x_embs, lengths

    def get_states(self, x):
        self.q_device.reset_states(x.shape[0])
        x_embs, lengths = self.get_embeddings(x)

        states = []
        for i in range(x.shape[1]):
            for j in range(self.reupload_count):
                self.encoder(self.q_device,
                             self.reupload_weights[j] * x_embs[:, i] + self.reupload_bias[j],
                             reset=False)
            if i != x.shape[1] - 1:
                self.recurrent_circuit(self.q_device, self.recurrent_params, reset=False)
            states.append(self.q_device.get_states_1d())

        states = torch.stack(states, dim=1)
        lengths = lengths.unsqueeze(2).repeat(1, 1, states.shape[2])
        return states.gather(1, lengths).squeeze()

    def get_exps(self, x):
        self.q_device.reset_states(x.shape[0])
        x_embs, lengths = self.get_embeddings(x)
        x_embs = self.dropout(x_embs)

        measures = []
        for i in range(x.shape[1]):
            for j in range(self.reupload_count):
                self.encoder(self.q_device,
                             self.reupload_weights[j] * x_embs[:, i] + self.reupload_bias[j],
                             reset=False)
            if i != x.shape[1] - 1:
                self.recurrent_circuit(self.q_device, self.recurrent_params, reset=False)
            measures.append(self.measure(self.q_device))

        measures = torch.stack(measures, dim=1)
        lengths = lengths.unsqueeze(2).repeat(1, 1, measures.shape[2])
        return measures.gather(1, lengths).squeeze()

    def draw(self, x):
        qc = tq2qiskit(self.q_device, self, x=x)
        qc.draw(output='text', filename='circ.txt')

    def forward(self, x):
        exps = self.get_exps(x) #.unsqueeze(-1)
        f = self.final(exps)
        return f


class RandomEmbedQRNNCaptum(RandomEmbedQRNN):
    def __init__(self,
                 vocab_size,
                 encoder_ansatz,
                 embed_dim=0,
                 encoder_layers=1,
                 reupload_count=1,
                 recurrent_ansatz=None,
                 recurrent_layers=1,
                 n_wires=4,
                 measure='all',
                 dropout_prob=0.0,
                 final='linear',
                 use_linear=False):
        super().__init__(vocab_size, encoder_ansatz, embed_dim, encoder_layers,
                         reupload_count, recurrent_ansatz, recurrent_layers,
                         n_wires, measure, dropout_prob, final)

        self.use_linear = use_linear

    def get_pre_embeddings(self, x):
        lengths = (x == 0).long().argmax(1, keepdim=True) - 1
        lengths[lengths == -1] = x.shape[1] - 1
        x_embs = self.embedding(x)
        return x_embs, lengths

    def get_embeddings(self, x):
        if self.use_linear:
            return self.get_pre_embeddings(x)
        else:
            return super().get_embeddings(x)

    def get_exps(self, x_embs, lengths):
        self.q_device.reset_states(x_embs.shape[0])

        measures = []
        for i in range(x_embs.shape[1]):
            for j in range(self.reupload_count):
                self.encoder(self.q_device,
                             self.reupload_weights[j] * x_embs[:, i] + self.reupload_bias[j],
                             reset=False)
            if i != x_embs.shape[1] - 1:
                self.recurrent_circuit(self.q_device, self.recurrent_params, reset=False)
            measures.append(self.measure(self.q_device))

        measures = torch.stack(measures, dim=1)
        lengths = lengths.unsqueeze(2).repeat(1, 1, measures.shape[2])
        return measures.gather(1, lengths).squeeze()

    def forward(self, x_embs, lengths):
        if self.use_linear:
            x_embs = self.embed_transform(x_embs)
            x_embs = torch.sigmoid(x_embs)
        exps = self.get_exps(x_embs, lengths) # .unsqueeze(-1)
        f = self.final(exps)
        f = torch.sigmoid(f)
        return f


class DrawRandomEmbedQRNN(tq.QuantumModule):
    def __init__(self, n_wires, encoder_ansatz, encoder_layers):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        encoder_dict, self.num_params = encoder_ansatz(n_wires=n_wires,
                                                       layers=encoder_layers)
        self.encoder = tq.GeneralEncoder(encoder_dict)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def draw(self, params):
        from torchquantum.plugins import tq2qiskit
        qc = tq2qiskit(self.q_device, self, x=params)
        qc.draw(output='text', filename='circ.txt')

    def forward(self, q_device, params):
        q_device.reset_states(params.shape[0])

        for i in range(params.shape[1]):
            self.encoder(q_device, params[:, i], reset=False)

        return self.measure(q_device)


class PositionEmbedQRNN(QRNN):
    def __init__(self,
                 max_len,
                 classical_embed_dim,
                 encoder_ansatz,
                 encoder_layers=1,
                 reupload_count=1,
                 recurrent_ansatz=None,
                 recurrent_layers=1,
                 n_wires=4,
                 measure='all',
                 dropout_prob=0.0,
                 final='linear'):
        super().__init__(encoder_ansatz, encoder_layers, reupload_count,
                         recurrent_ansatz, recurrent_layers, n_wires, measure,
                         dropout_prob, final)

        self.position_embed = nn.Parameter(torch.randn((max_len, classical_embed_dim, self.num_params)))
        self.end_token_params = nn.Parameter(torch.rand(self.num_params))

    def forward(self, x):
        self.q_device.reset_states(x.shape[0])

        x_embs = torch.einsum('bld,ldn->bln', x, self.position_embed)
        x_embs = torch.sigmoid(x_embs)
        x_embs = self.dropout(x_embs)
        for i in range(x_embs.shape[1]):
            for j in range(self.reupload_count):
                self.encoder(self.q_device,
                             self.reupload_weights[j] * x_embs[:, i] + self.reupload_bias[j],
                             reset=False)

            self.recurrent_circuit(self.q_device, self.recurrent_params, reset=False)

        self.encoder(self.q_device, self.end_token_params.repeat(x.shape[0], 1), reset=False)

        out = self.measure(self.q_device)
        out = self.final(out)

        return out
