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

# 主模型
class ConcreteAttentionModel(nn.Module):
    def __init__(
        self,
        image_feature_size=524288,
        text_feature_dim=18,          # 连续特征的维度
        feature_size_comp=512,
        feature_size_attn=256,
        dropout=True,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
        fusion_type='kron',
        # fusion_type='concat',
        use_bilinear=True,
        gate_hist=True,
        gate_text=False,     # 设为False后，文本特征不会再被图像特征影响
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

        # 注意力模块
        self.attention_survival_net = AttnNet(
            L=feature_size_comp, D=feature_size_attn, dropout=dropout, p_dropout_atn=p_dropout_atn
        )

        # 双模态注意力门控
        self.attn_modalities = Attn_Modality_Gated(
            gate_h1=gate_hist,
            gate_h2=gate_text,
            dim1_og=feature_size_comp,    # 图像分支：512
            dim2_og=feature_size_comp,    # 文本连续分支：512
            use_bilinear=use_bilinear,
            scale=1,
            p_dropout_fc=p_dropout_fc
        )

        # 融合后特征压缩
        if fusion_type == 'kron':
            # 文本分支仅为 feature_size_comp 维，融合后维度为 512 * 512 = 262144
            fusion_dim = feature_size_comp * feature_size_comp
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

    def forward(self, image_features, text_features, return_emb=False):
        """
        新增参数:
            return_emb: 若为 True，则在返回中额外提供 image_emb 和 text_emb
        """
        # 图像特征处理
        image_features = self.image_compression(image_features)  # 输入形状 (B, 9, 524288) -> (B, 9, 512)

        # 图像特征的注意力加权
        A_raw = self.attention_survival_net(image_features)   # A_raw shape: (B, 9, 1)
        image_features = A_raw * image_features                # image_features shape: (B, 9, 512)
        # 聚合多个 patch 的图像特征：取平均或求和
        image_features = image_features.mean(dim=1)             # image_features shape: (B, 512)

        # 连续值文本特征处理
        text_cont = self.text_continuous_layer(text_features)
        combined_text = text_cont

        # 双模态门控融合，只传入图像和文本连续特征
        image_features, combined_text = self.attn_modalities(image_features, combined_text)

        # 返回融合前图像文本特征：image_emb, text_emb 均是 (B, 512)
        image_emb = image_features
        text_emb = combined_text

        # 特征融合
        if self.fusion_type == 'kron':
            # print("使用的融合方式是：kron")
            fused = torch.stack(
                [torch.kron(image_features[i], combined_text[i])
                 for i in range(image_features.size(0))], 
                 dim=0
            )
        elif self.fusion_type == 'concat':
            # print("使用的融合方式是：concat")
            fused = torch.cat([image_features, combined_text], dim=-1)
        else:
            raise ValueError("Unsupported fusion type")

        # 融合特征后处理
        fused = self.post_fusion_layer(fused)

        # 输出预测
        output = self.classifier(fused)

        if return_emb:
            return output, image_emb, text_emb
        else:
            return output
