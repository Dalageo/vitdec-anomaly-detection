import torch
import numpy as np
import torch.nn as nn


# ---------------------------------------
# Load Vision Transformer Encoder Weights
# ---------------------------------------
def load_weights(model, checkpoint_path):
    weights = np.load(checkpoint_path, allow_pickle=True)

    def _n2p(w, transpose=False):
        """Convert numpy array to PyTorch tensor, with optional transpose."""
        if transpose:
            w = w.T
        return torch.from_numpy(w)

    # Load weights into the model
    # Embedding layer weights
    model.patch_embed.proj.weight.data = _n2p(weights['embedding/kernel'], transpose=True)
    model.patch_embed.proj.bias.data = _n2p(weights['embedding/bias'])

    # CLS token and position embeddings
    model.cls_token.data = _n2p(weights['cls'], transpose=False)
    model.pos_embed.data = _n2p(weights['Transformer/posembed_input/pos_embedding'])

    # Transformer blocks
    for i, block in enumerate(model.blocks):
        block_prefix = f"Transformer/encoderblock_{i}/"
        # LayerNorms within each block
        block.norm1.weight.data = _n2p(weights[f"{block_prefix}LayerNorm_0/scale"])
        block.norm1.bias.data = _n2p(weights[f"{block_prefix}LayerNorm_0/bias"])
        block.norm2.weight.data = _n2p(weights[f"{block_prefix}LayerNorm_2/scale"])
        block.norm2.bias.data = _n2p(weights[f"{block_prefix}LayerNorm_2/bias"])
        
        # Multihead attention weights
        qkv = _n2p(weights[f"{block_prefix}MultiHeadDotProductAttention_1/query/kernel"], transpose=True)
        qkv = torch.cat([qkv, _n2p(weights[f"{block_prefix}MultiHeadDotProductAttention_1/key/kernel"], transpose=True), 
                         _n2p(weights[f"{block_prefix}MultiHeadDotProductAttention_1/value/kernel"], transpose=True)], dim=0)
        block.attn.qkv.weight.data = qkv.view(block.attn.qkv.weight.shape)
        
        # MLP weights
        block.mlp.fc1.weight.data = _n2p(weights[f"{block_prefix}MlpBlock_3/Dense_0/kernel"], transpose=True)
        block.mlp.fc1.bias.data = _n2p(weights[f"{block_prefix}MlpBlock_3/Dense_0/bias"])
        block.mlp.fc2.weight.data = _n2p(weights[f"{block_prefix}MlpBlock_3/Dense_1/kernel"], transpose=True)
        block.mlp.fc2.bias.data = _n2p(weights[f"{block_prefix}MlpBlock_3/Dense_1/bias"])

    # Head (classifier) layer
    if 'head/kernel' in weights and isinstance(model.head, nn.Linear) and model.head.weight.shape == _n2p(weights['head/kernel'], transpose=True).shape:
        model.head.weight.data = _n2p(weights['head/kernel'], transpose=True)
        model.head.bias.data = _n2p(weights['head/bias'])

    # Final LayerNorm
    if 'Transformer/encoder_norm/scale' in weights:
        model.norm.weight.data = _n2p(weights['Transformer/encoder_norm/scale'])
        model.norm.bias.data = _n2p(weights['Transformer/encoder_norm/bias'])



# --------------------------
# Initialize Decoder Weights
# --------------------------
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    module.weight.data.fill_(1)
                if module.bias is not None:
                    module.bias.data.zero_()