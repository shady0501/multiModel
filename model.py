import torch
import torch.nn as nn
import torch.nn.functional as F

# 基于注意力机制的神经网络模块
class AttnNet(nn.Module):
    def __init__(self, L, D, dropout=False, p_dropout_atn=0.25, n_classes=1):
        super(AttnNet, self).__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_c = nn.Linear(D, n_classes)
        self.dropout = nn.Dropout(p_dropout_atn) if dropout else nn.Identity()

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)
        A = F.softmax(A, dim=1)
        return A

# 双模态注意力门控模块
class Attn_Modality_Gated(nn.Module):
    def __init__(self, gate_h1, gate_h2, dim1_og, dim2_og, use_bilinear=True, scale=1, p_dropout_fc=0.25):
        super(Attn_Modality_Gated, self).__init__()
        self.gate_h1 = gate_h1
        self.gate_h2 = gate_h2
        self.use_bilinear = use_bilinear
        
        dim1, dim2 = dim1_og // scale, dim2_og // scale

        if gate_h1:
            self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
            self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
            self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=p_dropout_fc))
        else:
            self.linear_h1, self.linear_o1 = nn.Identity(), nn.Identity()

        if gate_h2:
            self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
            self.linear_z2 = nn.Bilinear(dim2_og, dim1_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
            self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=p_dropout_fc))
        else:
            self.linear_h2, self.linear_o2 = nn.Identity(), nn.Identity()

    def forward(self, x1, x2):
        if self.gate_h1:
            h1 = self.linear_h1(x1)
            z1 = self.linear_z1(x1, x2) if self.use_bilinear else self.linear_z1(torch.cat((x1, x2), dim=-1))
            o1 = self.linear_o1(torch.sigmoid(z1) * h1)
        else:
            h1 = self.linear_h1(x1)
            o1 = self.linear_o1(h1)

        if self.gate_h2:
            h2 = self.linear_h2(x2)
            z2 = self.linear_z2(x2, x1) if self.use_bilinear else self.linear_z2(torch.cat((x1, x2), dim=-1))
            o2 = self.linear_o2(torch.sigmoid(z2) * h2)
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
    def __init__(self, taxonomy_in, embedding_dim, depth=1, act_fct='relu', dropout=True, p_dropout=0.25):
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
class ConcreteAttentionModel(nn.Module):
    def __init__(
        self,
        image_feature_size=1024,
        text_feature_dim=18,
        categorical_dims=[10, 3, 3, 3, 3, 10],
        embedding_dim=128,
        feature_size_comp=512,
        feature_size_attn=256,
        dropout=True,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
        fusion_type='kron',
        use_bilinear=True,
        gate_hist=True,
        gate_text=True,
    ):
        super(ConcreteAttentionModel, self).__init__()
        self.fusion_type = fusion_type

        # 图像特征压缩
        self.image_compression = nn.Sequential(
            nn.Linear(image_feature_size, feature_size_comp),
            nn.ReLU(),
            nn.Dropout(p_dropout_fc)
        )

        # 连续值特征处理
        self.text_continuous_layer = nn.Sequential(
            nn.Linear(text_feature_dim, feature_size_comp),
            nn.ReLU(),
            nn.Dropout(p_dropout_fc)
        )

        # 离散值嵌入处理
        self.categorical_encodings = nn.ModuleList([
            Categorical_encoding(dim, embedding_dim, dropout=dropout, p_dropout=p_dropout_fc) for dim in categorical_dims
        ])
        self.text_cat_compression = nn.Sequential(
            nn.Linear(len(categorical_dims) * embedding_dim, feature_size_comp),
            nn.ReLU(),
            nn.Dropout(p_dropout_fc)
        )

        # 注意力模块
        self.attention_survival_net = AttnNet(
            L=feature_size_comp, D=feature_size_attn, dropout=dropout, p_dropout_atn=p_dropout_atn
        )

        # 双模态注意力门控
        self.attn_modalities = Attn_Modality_Gated(
            gate_h1=gate_hist, gate_h2=gate_text, dim1_og=feature_size_comp, dim2_og=feature_size_comp,
            use_bilinear=use_bilinear, p_dropout_fc=p_dropout_fc
        )

        # 融合后特征压缩
        if fusion_type == 'kron':
            fusion_dim = feature_size_comp ** 2
        elif fusion_type == 'concat':
            fusion_dim = feature_size_comp * 2
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.post_fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, feature_size_comp),
            nn.ReLU(),
            nn.Dropout(p_dropout_fc)
        )

        # 回归头
        self.classifier = nn.Linear(feature_size_comp, 1)

    def forward(self, image_features, text_features, categorical_features):
        # 图像特征处理
        image_features = self.image_compression(image_features)

        # 连续值文本特征处理
        text_cont = self.text_continuous_layer(text_features)

        # 离散值处理
        text_cat = [enc(cat_feat) for enc, cat_feat in zip(self.categorical_encodings, categorical_features)]
        text_cat = torch.cat(text_cat, dim=-1)
        text_cat = self.text_cat_compression(text_cat)

        # 图像特征的注意力加权
        A_raw = self.attention_survival_net(image_features)
        image_features = A_raw * image_features

        # 双模态注意力门控
        image_features, text_features = self.attn_modalities(image_features, torch.cat([text_cont, text_cat], dim=-1))

        # 特征融合
        if self.fusion_type == 'kron':
            fused = torch.kron(image_features, text_features)
        elif self.fusion_type == 'concat':
            fused = torch.cat([image_features, text_features], dim=-1)
        else:
            raise ValueError("Unsupported fusion type")

        # 融合特征后处理
        fused = self.post_fusion_layer(fused)

        # 输出预测
        output = self.classifier(fused)
        return output
