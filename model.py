import torch
import torch.nn as nn
import torch.nn.functional as F

# 基于注意力机制的神经网络模块
class AttnNet(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, p_dropout_atn=0.25, n_classes=1):
        super(AttnNet, self).__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        if dropout:
            self.attention_a.append(nn.Dropout(p_dropout_atn))
            self.attention_b.append(nn.Dropout(p_dropout_atn))
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A

# 双模态注意力门控模块
class Attn_Modality_Gated(nn.Module):
    def __init__(self, gate_h1, gate_h2, dim1_og, dim2_og, use_bilinear=[True, True], scale=[1, 1], p_dropout_fc=0.25):
        super(Attn_Modality_Gated, self).__init__()
        self.gate_h1 = gate_h1
        self.gate_h2 = gate_h2
        self.use_bilinear = use_bilinear

        dim1, dim2 = dim1_og // scale[0], dim2_og // scale[1]

        if self.gate_h1:
            self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
            self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if self.use_bilinear[0] else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
            self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=p_dropout_fc))
        else:
            self.linear_h1, self.linear_o1 = nn.Identity(), nn.Identity()

        if self.gate_h2:
            self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
            self.linear_z2 = nn.Bilinear(dim2_og, dim1_og, dim2) if self.use_bilinear[1] else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
            self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=p_dropout_fc))
        else:
            self.linear_h2, self.linear_o2 = nn.Identity(), nn.Identity()

    def forward(self, x1, x2):
        if self.gate_h1:
            h1 = self.linear_h1(x1)
            z1 = self.linear_z1(x1, x2) if self.use_bilinear[0] else self.linear_z1(torch.cat((x1, x2), dim=-1))
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            h1 = self.linear_h1(x1)
            o1 = self.linear_o1(h1)

        if self.gate_h2:
            h2 = self.linear_h2(x2)
            z2 = self.linear_z2(x2, x1) if self.use_bilinear[1] else self.linear_z2(torch.cat((x1, x2), dim=-1))
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            h2 = self.linear_h2(x2)
            o2 = self.linear_o2(h2)

        return o1, o2

# 全连接层模块
class FC_block(nn.Module):
    def __init__(self, dim_in, dim_out, act_layer=nn.ReLU, dropout=True, p_dropout_fc=0.25):
        super(FC_block, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        self.act = act_layer()
        self.drop = nn.Dropout(p_dropout_fc) if dropout else nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x

# 分类变量编码模块
class Categorical_encoding(nn.Module):
    def __init__(self, taxonomy_in=3, embedding_dim=128, depth=1, act_fct='relu', dropout=True, p_dropout=0.25):
        super(Categorical_encoding, self).__init__()

        act_fcts = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'selu': nn.SELU(),
        }
        dropout_module = nn.AlphaDropout(p_dropout) if act_fct == 'selu' else nn.Dropout(p_dropout)

        self.embedding = nn.Embedding(taxonomy_in, embedding_dim)

        fc_layers = []
        for d in range(depth):
            fc_layers.append(nn.Linear(embedding_dim // (2**d), embedding_dim // (2**(d + 1))))
            fc_layers.append(dropout_module if dropout else nn.Identity())
            fc_layers.append(act_fcts[act_fct])

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc_layers(x)
        return x

# 主模型
class HECTOR(nn.Module):
    def __init__(
        self,
        input_feature_size=1024,  # 图像特征维度
        input_text_size=512,      # 文本特征维度
        input_text_cat_size=4,    # 文本分类变量维度
        precompression_layer=True,
        feature_size_comp=512,
        feature_size_attn=256,
        postcompression_layer=True,
        feature_size_comp_post=128,
        dropout=True,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
        fusion_type='kron',
        use_bilinear=[True, True],
        gate_hist=False,
        gate_text=False,
        scale=[1, 1],
    ):
        super(HECTOR, self).__init__()
        self.fusion_type = fusion_type
        self.use_bilinear = use_bilinear
        self.gate_hist = gate_hist
        self.gate_text = gate_text

        # 图像特征压缩
        if precompression_layer:
            self.compression_layer = nn.Sequential(
                FC_block(input_feature_size, feature_size_comp * 4, p_dropout_fc=p_dropout_fc),
                FC_block(feature_size_comp * 4, feature_size_comp * 2, p_dropout_fc=p_dropout_fc),
                FC_block(feature_size_comp * 2, feature_size_comp, p_dropout_fc=p_dropout_fc),
            )
            dim_post_compression = feature_size_comp
        else:
            self.compression_layer = nn.Identity()
            dim_post_compression = input_feature_size

        # 文本特征压缩
        self.text_compression_layer = nn.Sequential(
            FC_block(input_text_size, feature_size_comp, p_dropout_fc=p_dropout_fc),
        )

        # 文本分类变量编码
        self.text_cat_encoding = Categorical_encoding(
            taxonomy_in=input_text_cat_size,
            embedding_dim=feature_size_comp,
            depth=1,
            act_fct='relu',
            dropout=dropout,
            p_dropout=p_dropout_fc,
        )

        # 注意力机制
        self.attention_survival_net = AttnNet(
            L=dim_post_compression,
            D=feature_size_attn,
            dropout=dropout,
            p_dropout_atn=p_dropout_atn,
            n_classes=1,
        )

        # 双模态注意力门控
        self.attn_modalities = Attn_Modality_Gated(
            gate_h1=self.gate_hist,
            gate_h2=self.gate_text,
            dim1_og=dim_post_compression,
            dim2_og=feature_size_comp,
            use_bilinear=self.use_bilinear,
            scale=scale,
        )

        # 图像特征后压缩
        dim_post_compression = dim_post_compression // scale[0] if self.gate_hist else dim_post_compression
        self.post_compression_layer_he = FC_block(dim_post_compression, dim_post_compression // 2, p_dropout_fc=p_dropout_fc)
        dim_post_compression = dim_post_compression // 2

        # 融合后的特征后压缩
        dim1, dim2 = dim_post_compression, feature_size_comp // scale[1] if self.gate_text else feature_size_comp
        if self.fusion_type == 'bilinear':
            head_size_in = (dim1 + 1) * (dim2 + 1)
        elif self.fusion_type == 'kron':
            head_size_in = dim1 * dim2
        elif self.fusion_type == 'concat':
            head_size_in = dim1 + dim2

        self.post_compression_layer = nn.Sequential(
            FC_block(head_size_in, feature_size_comp_post * 2, p_dropout_fc=p_dropout_fc),
            FC_block(feature_size_comp_post * 2, feature_size_comp_post, p_dropout_fc=p_dropout_fc),
        )

        # 回归头
        self.classifier = nn.Linear(feature_size_comp_post, 1)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward_attention(self, h):
        A_ = self.attention_survival_net(h)
        A_raw = torch.transpose(A_, 1, 0)
        A = F.softmax(A_raw, dim=-1)
        return A_raw, A

    def forward_fusion(self, h1, h2):
        if self.fusion_type == 'bilinear':
            h1 = torch.cat((h1, torch.ones(1, 1, dtype=torch.float, device=h1.device)), -1)
            h2 = torch.cat((h2, torch.ones(1, 1, dtype=torch.float, device=h2.device)), -1)
            return torch.kron(h1, h2)
        elif self.fusion_type == 'kron':
            return torch.kron(h1, h2)
        elif self.fusion_type == 'concat':
            return torch.cat([h1, h2], dim=-1)
        else:
            raise ValueError("Fusion type not implemented")

    def forward(self, h, text, text_cat):
        # 图像特征压缩
        h = self.compression_layer(h)

        # 注意力机制
        A_raw, A = self.forward_attention(h)
        h_hist = A @ h

        # 文本特征压缩
        text = self.text_compression_layer(text)

        # 文本分类变量编码
        text_cat = self.text_cat_encoding(text_cat)

        # 将文本特征和分类变量特征拼接
        text_combined = torch.cat([text, text_cat], dim=-1)

        # 双模态注意力门控
        h_hist, text_combined = self.attn_modalities(h_hist, text_combined)

        # 图像特征后压缩
        h_hist = self.post_compression_layer_he(h_hist)

        # 特征融合
        m = self.forward_fusion(h_hist, text_combined)

        # 融合特征后压缩
        m = self.post_compression_layer(m)

        # 回归头
        output = self.classifier(m)
        return output