import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from app.utils.log_utils import LoggerConfig
from app.config import IMG_SIZE, VIT_WEIGHTS_PATH, DEVICE
from app.model.weights import load_weights, initialize_weights

logger = LoggerConfig().get_logger(__name__)

# -------------------------
# Tuple Conversion Function
# -------------------------
def to_2tuple(x):
    """Convert a single value to a 2-tuple, or return the tuple if it's already one."""
    return (x, x) if not isinstance(x, tuple) else x

# -----------------------
# MLP Module with Dropout
# -----------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
# ----------------------
# Patch Embedding Layer
# ----------------------
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding"""
    
    def __init__(self, img_size=384, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # Final projection to embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x) 
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    
    
# ----------------
# Attention Block
# ----------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------
# Drop Path
# ---------
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with any number of dimensions, not just 4D tensors
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
    
    
# ----------------
# Transformer Block
# ----------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
# ------------------------
# Vision Transformer Model
# ------------------------
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=2, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
            self.num_features = embed_dim

        # Linear layer for classification
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x) # Input image is divided into patchs and each patch is converted to an embedding
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # Expand the cls token to match the batch size 
        # the cls_token becomes a tensor with shape [batch_size, 1, embed_dim] for each batch
        x = torch.cat((cls_token, x), dim=1)  # Concatenate the class token in the beggining of the sequence 
        x = self.pos_drop(x + self.pos_embed) # Add the positional embeddings
        x = self.blocks(x) # Pass through the blocks to extract features
        x = self.norm(x) # Normalize the final output
        return x

    def forward(self, x, include_cls_token=False):
        features = self.forward_features(x)  # Extract features from transformer blocks
        # Feature shape [16, 577, 768] (576 patches + 1 CLS token)
        if include_cls_token:
            cls_token_output = features[:, 0]  # Extract the CLS token for classification [2, 768]
            features_output = features[:, 1:]  # Extract the remaining features for other tasks like reconstruction [16, 576, 768] (remaining patches)
            logits = self.head(cls_token_output)  # Pass the cls through the head 
            return logits, features_output  # Return logits and the features
        else:
            features_output = features[:, 1:]  # Use only the non-CLS token features
            return features_output  # For unsupervised tasks where classification isn't needed


# -------------
# Decoder Model
# -------------
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.img_size = IMG_SIZE
        self.dec_block1 = nn.Sequential(
            # Input: [1, 768, 24, 24]
            nn.ConvTranspose2d(768, 384, (3, 3)), # Expanding informations to larger size (upsample increasing width and height) 
            nn.InstanceNorm2d(384), # Normalizing the output feature maps
            nn.ReLU(True)
            # Output: [1, 384, 26, 26]
        )
        
        self.dec_block2 = nn.Sequential(
            # Input: [1, 384, 26, 26]
            nn.ConvTranspose2d(384, 192, (3, 3)), 
            nn.InstanceNorm2d(192),
            nn.ReLU(True)
            # Output: [1, 192, 28, 28]
        )
        
        self.dec_block3 = nn.Sequential(
            # Input: [1, 192, 28, 28]
            nn.ConvTranspose2d(192, 96, (3, 3)), 
            nn.InstanceNorm2d(96),
            nn.ReLU(True)
            # Output: [1, 96, 30, 30]
        )

        self.dec_block4 = nn.Sequential(
            # Input: [1, 96, 30, 30]
            nn.ConvTranspose2d(96, 48, (3, 3)),
            nn.InstanceNorm2d(48),
            nn.ReLU(True)
            # Output: [1, 48, 32, 32]
        )

        self.dec_block5 = nn.Sequential(
            # Input: [1, 48, 32, 32]
            nn.ConvTranspose2d(48, 24, (3, 3)),
            nn.InstanceNorm2d(24),
            nn.ReLU(True)
            # Output: [1, 24, 34, 34]
        )

        self.dec_block6 = nn.Sequential(
            # Input: [1, 24, 34, 34]
            nn.ConvTranspose2d(24, 12, (3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(12),
            nn.ReLU(True)
            # Output: [1, 12, 69, 69]
        )

        self.dec_block7 = nn.Sequential(
            # Input: [1, 12, 69, 69]
            nn.ConvTranspose2d(12, 6, (3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(6),
            nn.ReLU(True)
            # Output: [1, 6, 139, 139]
        )

        self.dec_block8 = nn.Sequential(
            # Input: [1, 6, 139, 139]
            nn.ConvTranspose2d(6, 3, (3, 3), stride=2, padding=1),
            nn.InstanceNorm2d(3),
            nn.ReLU(True)
            # Output: [1, 3, 279, 279]
        )
        
        # Input: [1, 3, 279, 279]
        self.up = nn.UpsamplingBilinear2d((self.img_size, self.img_size)) # A final upsample to ensure that the output matches the desire dimension (384,384)
        # Output: [1, 3, 384, 384]
        self.tanh = nn.Tanh() # Scale the output to be -1 to 1

    def forward(self,x):
        # Transform vit embeddings back to a spatial format
        batch_size = x.shape[0]   # Batch Size --> 16
        num_patches = x.shape[1]  # Number of patches --> 576 since CLS is removed
        feature_dim = x.shape[2]  # Embedding dimensions --> 768
    
        # 576 patches are reshaped to (24, 24) feature map (576^0.5)
        num_patches_per_side = int((num_patches)**0.5)
        # Transpose [16, 768, 576^0.5, 576^0.5]
        # Reshape [16, 768, 576^0.5, 576^0.5]
        out = x.transpose(1, 2).reshape(batch_size, feature_dim, num_patches_per_side, num_patches_per_side)
        
        out = self.dec_block1(out)
        out = self.dec_block2(out)
        out = self.dec_block3(out)
        out = self.dec_block4(out)
        out = self.dec_block5(out)
        out = self.dec_block6(out)
        out = self.dec_block7(out)
        out = self.dec_block8(out)

        out = self.up(out)
        out = self.tanh(out)
        
        return out


# ----------------------------------
# Vision Transformer - Decoder Model
# ----------------------------------
class ViTDecoder(nn.Module):
    def __init__(self, vit_encoder, decoder):
        super(ViTDecoder, self).__init__()
        self.vit_encoder = vit_encoder
        self.decoder = decoder
        
        # Load the model weights
        self.weights = load_weights(vit_encoder, VIT_WEIGHTS_PATH)
        
    def forward(self, x, return_logits=False, return_reconstruction=True):
        """Forward pass of the ViTDecoder module."""
        
        # Pass the input x to the transformer(vit_encoder) and optionally retrieve the logits
        if return_logits:
            cls_token_logits, features_output = self.vit_encoder(x, include_cls_token=True)
        else:
            features_output = self.vit_encoder(x, include_cls_token=False)
            cls_token_logits = None
        
        # Pass the features through the decoder to reconstruct the output, if required
        if return_reconstruction:
            reconstructed_output = self.decoder(features_output)
        else:
            reconstructed_output = None

        # Return the requested outputs
        return cls_token_logits, reconstructed_output
    
    
# -----------------------------
# Get ViTDecoder Model Function
# -----------------------------
def get_vitdec(output_test: bool=False):
    # Initialize the Vision Transformer
    vit_encoder = VisionTransformer(
        img_size=IMG_SIZE,  
        patch_size=16,
        in_chans=3,
        num_classes=1000,  
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        # drop_rate=0.5, # Applies to pos_drop and potentially MLP blocks if passed correctly
        # attn_drop_rate=0.5,  # This needs to be passed to the Block for attention dropout
    )

    # Change the classifier head for the classification task
    num_classes = 2
    vit_encoder.head = nn.Linear(vit_encoder.head.in_features, num_classes) 
    # Move it to device
    vit_encoder.to(DEVICE)
    
    # Initialize the Decoder
    decoder = Decoder() 
    # Set its weights 
    initialize_weights(decoder)
    # Move it to device
    decoder.to(DEVICE)
        
    # If the test argument is set, test the model output
    if output_test:
        tester = ViTDecTester(vit_encoder, decoder, DEVICE)
        tester.test_models(IMG_SIZE)
        
    # Combine the Vision Transformer and Decoder into a single model, and transfer it to the device
    vit_dec = ViTDecoder(vit_encoder, decoder).to(DEVICE)
    
    # Return the model
    return vit_dec


# -----------------------
# Model Output Test Class
# -----------------------
class ViTDecTester:
    def __init__(self, vit_model, decoder_model):
        self.vit_model = vit_model
        self.decoder_model = decoder_model
        self.img_size = IMG_SIZE
        self.device = DEVICE

    # Generate a dummy input tensor based on the image resolution and model input format
    def generate_dummy_input(self):
        dummy_input = torch.randn(1, 3, self.img_size, self.img_size)
        return dummy_input.to(self.device)

    # Test both the Vit and decoder model
    def test_models(self):
        dummy_input = self.generate_dummy_input(self.img_size)

        # Pass the dummy input through the vision transformer model
        logits, features_output = self.vit_model(dummy_input, include_cls_token=True)
        # Pass the features output through the decoder model
        decoded_image = self.decoder_model(features_output)

        logger.info("---------- Vision Transformer Test Output ----------")
        logger.info(f"Classification Logits Output Shape (for Classification): {logits.shape}")
        logger.info(f"Features Output Shape (for Decoder Input): {features_output.shape}")
        logger.info("---------- Decoder Test Output ----------")
        logger.info("Output Image Shape (Reconstructed from Features):", decoded_image.shape)
 


