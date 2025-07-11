import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from WavKAN import KAN
#from efficient_kan import KAN
import pytorch_lightning as pl
from torchinfo import summary
from globals import LR, BETAS, WEIGHT_DECAY, DROP_PROB


# Encoder modules ----------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
class ConvEncoder1D(nn.Module):
    """
    1D Convolutional encoder with optional attention mechanism and adaptive pooling.

    Args:
        in_chs (int): Number of input channels.
        out_chs (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        act_fun (nn.Module): Activation function.
        out_size (int): Output feature size after pooling.
        attention_type (str): Type of attention mechanism ('posenc', 'se', 'convattn').
        attention_params (dict): Parameters for the attention mechanism.
    """
    def __init__(self, in_chs=None, out_chs=128, kernel_size=3, act_fun=nn.ReLU(), out_size=16, 
                 attention_type=None, attention_params=None):
        super().__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.kernel_size = kernel_size
        self.attention_type = attention_type
        self.attention_params = attention_params or {}
        self.act_fun = act_fun
        self.out_size = out_size

        # Encoder layers will be built lazily at first forward pass.
        self._is_built = False

    def build(self, x):
        """
        Lazily builds the encoder layers based on the input tensor shape.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, time].
        """
        in_chs = x.shape[1]

        # Define convolution, activation, pooling, and flatten layers
        self.conv = nn.Conv1d(in_channels=in_chs,
                              out_channels=self.out_chs,
                              kernel_size=self.kernel_size,
                              padding=self.kernel_size // 2)
        self.bn = nn.BatchNorm1d(self.out_chs)
        self.activation = self.act_fun
        self.pool = nn.AdaptiveAvgPool1d(self.out_size)
        self.flatten = nn.Flatten()

        # Select attention mechanism if specified (using output channels of conv)
        if self.attention_type == 'posenc':
            self.attn = PositionalEncoding1D(self.out_chs, **self.attention_params)
        elif self.attention_type == 'se':
            self.attn = SqueezeExcite1D(self.out_chs, **self.attention_params)
        elif self.attention_type == 'convattn':
            self.attn = ConvAttention1D(self.out_chs, **self.attention_params)
        else:
            self.attn = nn.Identity()

        self._is_built = True
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, time].
        Returns:
            torch.Tensor: Encoded output tensor after conv, attention, activation, pooling, and flattening.
        """
        if not self._is_built:
            self.build(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.attn:
            x = self.attn(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.flatten(x)
        
        return x


class ConvEncoder2D(nn.Module):
    """
    2D Convolutional encoder with optional attention mechanism and adaptive pooling.

    Args:
        in_chs (int): Number of input channels.
        out_chs (int): Number of output channels.
        kernel_size (tuple): Size of the convolution kernel.
        act_fun (nn.Module): Activation function.
        out_size (tuple): Output feature size after pooling.
        attention_type (str): Type of attention mechanism ('axial', 'mhsa').
        attention_params (dict): Parameters for the attention mechanism.
    """
    def __init__(self, in_chs=None, out_chs=128, kernel_size=(3, 3), act_fun=nn.ReLU(), out_size=(8, 8), 
                 attention_type=None, attention_params=None):
        super().__init__()
        self.out_chs = out_chs
        self.kernel_size = kernel_size
        self.attention_type = attention_type
        self.attention_params = attention_params or {}
        self.act_fun = act_fun
        self.out_size = out_size
        
        self._is_built = False

    def build(self, x):
        """
        Lazily builds the encoder layers based on the input tensor shape.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width].
        """
        in_chs = x.shape[1]

        # Downsampling (before MHSA) to reduce module's memory usage
        if self.attention_type == 'mhsa':
            self.down = nn.AdaptiveAvgPool2d((32, 32))
        else:
            self.down = nn.Identity()

        # Select attention mechanism if specified
        if self.attention_type == 'axial':
            self.attn = AxialAttention2D(in_chs, **self.attention_params)
        elif self.attention_type == 'mhsa':
            self.attn = MHSA2D(in_chs, **self.attention_params)
        else:
            self.attn = nn.Identity()

        # Define convolution, activation, pooling, and flatten layers
        self.conv = nn.Conv2d(in_channels=in_chs,
                              out_channels=self.out_chs,
                              kernel_size=self.kernel_size,
                              padding='same')
        self.bn = nn.BatchNorm2d(self.out_chs)
        self.activation = self.act_fun
        self.pool = nn.AdaptiveAvgPool2d(self.out_size)
        self.flatten = nn.Flatten()

        self._is_built = True
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width].
        Returns:
            torch.Tensor: Encoded output tensor after attention, conv, activation, pooling, and flattening.
        """
        if not self._is_built:
            self.build(x)
        x = self.down(x)
        x = self.attn(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.flatten(x)
        
        return x


# Attention modules --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
class PositionalEncoding1D(nn.Module):
    """
    Adds sinusoidal positional encoding to 1D signals, generated on-the-fly based on the input length.

    Args:
        channels (int): Number of input feature channels.
    """
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, time].
        Returns:
            torch.Tensor: Tensor of same shape as input with positional encoding added.
        """
        B, C, T = x.shape
        device = x.device

        position = torch.arange(0, T, device=device).unsqueeze(1).float()  # [T, 1]
        div_term = torch.exp(torch.arange(0, C, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / C))
        pe = torch.zeros(C, T, device=device)
        pe[0::2, :] = torch.sin(position * div_term).T
        pe[1::2, :] = torch.cos(position * div_term).T
        pe = pe.unsqueeze(0)  # [1, C, T]
        
        return x + pe


class SqueezeExcite1D(nn.Module):
    """
    Implements a Squeeze-and-Excitation (SE) block for 1D inputs, which adaptively 
    recalibrates channel-wise feature responses.

    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for the bottleneck in the SE block.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channels, reduced_channels),
                                nn.ReLU(),
                                nn.Linear(reduced_channels, channels),
                                nn.Sigmoid())

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, time].
        Returns:
            torch.Tensor: Output tensor with channel-wise recalibration, same shape as input.
        """
        b, c, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        
        return x * y.expand_as(x)


class ConvAttention1D(nn.Module):
    """
    Depthwise separable convolutional attention block for 1D signals.
    Applies a depthwise convolution followed by a pointwise convolution to capture local channel interactions.

    Args:
        channels (int): Number of input channels.
        kernel_size (int): Size of the depthwise convolution kernel.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size,
                                   groups=channels, padding=kernel_size // 2)
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, time].
        Returns:
            torch.Tensor: Output tensor of same shape.
        """
        return self.pointwise(self.depthwise(x))


class AxialAttention2D(nn.Module):
    """
    Implements axial attention along height and width axes for 2D inputs.
    This mechanism applies self-attention separately along each axis to efficiently model long-range 
    dependencies in 2D feature maps.

    Args:
        in_chs (int): Number of input channels.
        heads (int): Number of attention heads.
        dim_head (int): Dimensionality of each attention head.
    """
    def __init__(self, in_chs, heads=4, dim_head=16):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv_h = nn.Conv1d(in_chs, inner_dim * 3, 1, bias=False)
        self.to_qkv_w = nn.Conv1d(in_chs, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, in_chs, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width].
        Returns:
            torch.Tensor: Output tensor of shape [batch, channels, height, width].
        """
        B, C, H, W = x.shape
        # Height axis attention
        x_h = x.permute(0, 3, 1, 2).reshape(B * W, C, H)
        qkv = self.to_qkv_h(x_h).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(B * W, self.heads, self.dim_head, H), qkv)
        dots = (q * k).sum(dim=2) / self.dim_head**0.5
        attn = dots.softmax(dim=-1)
        out = (attn.unsqueeze(2) * v).sum(dim=3)
        out = out.view(B, W, self.heads * self.dim_head).permute(0, 2, 1).contiguous().view(B, self.heads * self.dim_head, 1, W)
        out = out.expand(-1, -1, H, -1)

        return self.to_out(out)


class MHSA2D(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) layer over 2D feature maps.
    This module applies self-attention across spatial locations for each channel using 
    multiple attention heads.

    Args:
        in_chs (int): Number of input channels.
        heads (int): Number of attention heads.
        dim_head (int): Dimensionality of each attention head.
    """
    def __init__(self, in_chs, heads=4, dim_head=16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_qkv = nn.Conv2d(in_chs, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, in_chs, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width].
        Returns:
            torch.Tensor: Output tensor of same shape.
        """
        B, C, H, W = x.shape
        qkv = self.to_qkv(x).reshape(B, 3, self.heads, self.dim_head, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv
        dots = torch.matmul(q, k.transpose(-1, -2)) / self.dim_head ** 0.5
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        
        return self.to_out(out)


# KAN modules --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
class BPMKAN(nn.Module):
    """
    WaveletKAN module for BPM regression.

    Args:
        in_feats (int): Number of input features.
    """
    def __init__(self, in_feats):
        super().__init__()
        layers_hidden = [in_feats] + [1]
        self.kan = KAN(layers_hidden=layers_hidden, wavelet_type='mexican_hat')

    def forward(self, x, encoder: nn.Module = None):
        if encoder is not None:
            x = encoder(x)
        bpm_hat = self.kan(x)
        
        return bpm_hat


class PatternKAN(nn.Module):
    """
    WaveletKAN module for rhythmic pattern classification.

    Args:
        in_feats (int): Number of input features.
        out_probs (int): Number of output classes.
    """
    def __init__(self, in_feats, out_probs=3):
        super().__init__()
        layers = [in_feats] + [out_probs]
        self.kan = KAN(layers_hidden=layers, wavelet_type='mexican_hat')

    def forward(self, x, encoder: nn.Module = None):
        if encoder is not None:
            x = encoder(x)
        pattern_logits = self.kan(x)
        
        return pattern_logits


# MLP modules --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
class BPMFC(nn.Module):
    """
    Fully connected MLP for BPM regression

    Args:
        in_feats (int): Number of input features.
        hidden_dims (list of int): Sizes of hidden layers.
        dropout (float): Dropout rate.
        activation (callable): Activation function (e.g., nn.ReLU()).
    """
    def __init__(self, in_feats, hidden_dims=[64, 32], dropout=DROP_PROB, activation=nn.ReLU()):
        super().__init__()
        layers = []
        prev_dim = in_feats

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))  # Output layer for regression
        self.model = nn.Sequential(*layers)

    def forward(self, x, encoder: nn.Module = None):
        if encoder is not None:
            x = encoder(x)
        return self.model(x)


class PatternFC(nn.Module):
    """
    Fully connected MLP for rhythmic pattern classification

    Args:
        in_feats (int): Number of input features.
        hidden_dims (list of int): Sizes of hidden layers.
        out_probs (int): Number of output classes.
        dropout (float): Dropout rate.
        activation (callable): Activation function (e.g., nn.ReLU()).
    """
    def __init__(self, in_feats, hidden_dims=[64, 32], out_probs=3, dropout=0.2, activation=nn.ReLU()):
        super().__init__()
        layers = []
        prev_dim = in_feats

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, out_probs))  # Output layer for classification
        self.model = nn.Sequential(*layers)

    def forward(self, x, encoder: nn.Module = None):
        if encoder is not None:
            x = encoder(x)
        return self.model(x)
    


# LIGHTNING WRAPPERS -------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
def init_modules(config, dummy_input):
    """
    Initializes encoder and KAN head based on configuration and dummy input.

    Args:
        config (dict): Model configuration dictionary.
        dummy_input (torch.Tensor): Dummy input tensor to determine input shape.

    Returns:
        encoder (nn.Module): Initialized encoder module.
        kan_head (nn.Module): Initialized KAN head module.
    """
    # 1) ConvEncoder parameters
    enc_type = config["encoder"]["type"]
    params = config["encoder"]["params"]
    attn_type = config["encoder"].get("attention", {}).get("type", None)
    attn_params = config["encoder"].get("attention", {}).get("params", {})

    # Pre-processing 2D-ConvEncoder parameters
    if enc_type == "2D":
        if isinstance(params.get("kernel_size"), list) and all(isinstance(k, list) for k in params["kernel_size"]):
            params["kernel_size"] = [tuple(k) for k in params["kernel_size"]]
        if isinstance(params.get("out_size"), list):
            params["out_size"] = tuple(params["out_size"])

    # Initialize encoder
    if enc_type == "1D":
        encoder = ConvEncoder1D(**params, attention_type=attn_type, attention_params=attn_params)
    elif enc_type == "2D":
        encoder = ConvEncoder2D(**params, attention_type=attn_type, attention_params=attn_params)
    else:
        raise ValueError(f"Unsupported encoder type: {enc_type}")

    # Forward encoding pass to determine output feature size
    dummy_encoded = encoder(dummy_input)
    in_feats = dummy_encoded.detach().shape[1]

    # 2) KAN parameters
    kan_type = config["kan"]["type"]
    kan_config = config["kan"].get("kan_config", None)

    # Initialize KAN head
    if kan_type == "BPMKAN":
        kan_head = BPMKAN(in_feats=in_feats, kan_config=kan_config) \
            if kan_config else BPMKAN(in_feats=in_feats)
    elif kan_type == "PatternKAN":
        out_probs = config["kan"].get("out_probs", 3)
        kan_head = PatternKAN(in_feats=in_feats, out_probs=out_probs, kan_config=kan_config) \
            if kan_config else PatternKAN(in_feats=in_feats, out_probs=out_probs)
    else:
        raise ValueError(f"Unsupported KAN type: {kan_type}")

    return encoder, kan_head


class Model(pl.LightningModule):
    def __init__(self, encoder_module, kan_module, downstream_task='bpm', lr=LR, config=None):
        super().__init__()
        self.encoder = encoder_module
        self.kan = kan_module
        self.task = downstream_task
        self.lr = lr
        self.loss_name = config["loss"]["name"]
        
        # Model loss config
        if self.task == 'bpm':
            self.loss_beta = config["loss"].get("kwargs", {}).get('beta', 0.8)
        else:
            self.tolerance = config["loss"].get("kwargs", {}).get("tolerance", 0.3)
            self.use_smoothing = config["loss"].get("kwargs", {}).get("use_smoothing", False)
        self.loss_fn = self.get_loss_fn()
        
        # Logging Metrics assets
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.val_correct = 0
        self.val_total = 0
        self.test_correct = 0
        self.test_total = 0
        
        # Test results storage
        self.test_preds = []
        self.test_gts = []

        self.save_hyperparameters(ignore=['encoder', 'kan', 'encoder_module', 'kan_module'])

        if self.task not in ['bpm', 'pattern']:
            raise ValueError("Downstream task must be: 'bpm' or 'pattern' (recognition)")

    def get_loss_fn(self):
        if self.task == 'bpm':
            if self.loss_name == 'mse':
                return F.mse_loss
            elif self.loss_name == 'mae':
                return F.l1_loss
            elif self.loss_name == 'huber':
                return lambda preds, targets: F.smooth_l1_loss(preds, targets, beta=self.loss_beta)
            elif self.loss_name == 'tempo_ratio':
                def tempo_ratio_loss(preds, targets):
                    eps = 1e-8
                    pred_safe = preds.clamp(min=eps)
                    target_safe = targets.clamp(min=eps)
                    ratio1 = pred_safe / target_safe
                    ratio2 = target_safe / pred_safe
                    min_ratio = torch.min(ratio1, ratio2)
                    loss = 1.0 - min_ratio
                    
                    return torch.mean(loss)
                
                return tempo_ratio_loss
        
        elif self.task == 'pattern':
            def categorical_cross_entropy(logits, targets, tolerance=self.tolerance, use_smoothing=self.use_smoothing):
                num_classes = logits.size(1)
                if not use_smoothing or tolerance == 0.0:
                    return F.cross_entropy(logits, targets)
                smoothing_mask = torch.eye(num_classes, device=logits.device)
                for i in range(num_classes):
                    for j in range(num_classes):
                        if i in [0, 1]:
                            if i == j:
                                smoothing_mask[i, j] = 1.0 - tolerance
                            elif j in [0, 1] and j != i:
                                smoothing_mask[i, j] = tolerance
                            else:
                                smoothing_mask[i, j] = 0.0
                        else:
                            smoothing_mask[i, j] = 1.0 if i == j else 0.0
                probs = F.softmax(logits, dim=1)
                smoothed_probs = probs @ smoothing_mask.T
                log_probs_smooth = torch.log(smoothed_probs + 1e-8)
                loss = F.nll_loss(log_probs_smooth, targets)
                
                return loss
            return categorical_cross_entropy
        
        else:
            raise ValueError(f"Unsupported loss_type '{self.loss_name}'.")

    def forward(self, x):
        return self.kan(x, encoder=self.encoder)

    def training_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["label"]
        y_hat = self(x)
        if self.task == 'bpm':
            y_hat = y_hat.squeeze(-1)   # schiacciamento dell'ultima dimensione per avere shape [batch] --> da verificare
        loss = self.loss_fn(y_hat, y)
        self.train_losses.append(loss.detach())
        self.log('train_loss_step', loss, prog_bar=True)
        
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log('train_loss', avg_loss, prog_bar=True)
        self.train_losses.clear()

    def validation_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["label"]
        y_hat = self(x)
        if self.task == 'bpm':
            y_hat = y_hat.squeeze(-1)   # schiacciamento dell'ultima dimensione per avere shape [batch] --> da verificare
        loss = self.loss_fn(y_hat, y)
        self.val_losses.append(loss.detach())
        
        if self.task == 'pattern':
            preds = torch.argmax(y_hat, dim=1)
            self.val_correct += (preds == y).sum().item()
            self.val_total += y.size(0)
        
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        self.val_losses.clear()
        
        if self.task == 'pattern':
            acc = self.val_correct / self.val_total if self.val_total > 0 else 0.0
            self.log('val_acc', acc, prog_bar=True)
            self.val_correct = 0
            self.val_total = 0

    def test_step(self, batch, batch_idx):
        x = batch["input"]
        y = batch["label"]
        y_hat = self(x)
        if self.task == 'bpm':
            y_hat = y_hat.squeeze(-1)   # schiacciamento dell'ultima dimensione per avere shape [batch] --> da verificare
        loss = self.loss_fn(y_hat, y)
        self.test_losses.append(loss.detach())
        
        if self.task == 'pattern':
            preds = torch.argmax(y_hat, dim=1)
            self.test_correct += (preds == y).sum().item()
            self.test_total += y.size(0)
        self.test_preds.append(y_hat.detach().cpu())
        self.test_gts.append(y.detach().cpu())
        
        return loss

    def on_test_epoch_end(self):
        if len(self.test_preds) > 0 and len(self.test_gts) > 0:
            all_preds = torch.cat(self.test_preds, dim=0).cpu().numpy()
            all_gts = torch.cat(self.test_gts, dim=0).cpu().numpy()
            self.test_preds.clear()
            self.test_gts.clear()
            # Save for external use
            self.test_outputs_npz = {'predictions': all_preds,
                                     'targets': all_gts}

        avg_loss = torch.stack(self.test_losses).mean().item()
        self.log('test_loss', avg_loss, prog_bar=True)
        self.test_losses.clear()

        metrics = {'test_loss': avg_loss,
                   'task': self.task}

        if self.task == 'pattern':
            acc = self.test_correct / self.test_total if self.test_total > 0 else 0.0
            self.log('test_acc', acc, prog_bar=True)
            metrics['test_metric'] = acc
            self.test_correct = 0
            self.test_total = 0
        
        elif self.task == 'bpm':
            from torch.nn.functional import mse_loss
            all_preds_tensor = torch.tensor(all_preds)
            all_gts_tensor = torch.tensor(all_gts)
            metrics['test_metric'] = mse_loss(all_preds_tensor, all_gts_tensor).item()

        # Save for external access
        self.test_metrics_summary = metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)


# Test commands ------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import torch
    import yaml

    ################################################ TEST HP-SEARCH CONFIGS ################################################
    CONFIG_FOLDER = "./experiments/configs/pattern"                     # or "bpm"

    # Create dummy input tensors for 1D and 2D encoders testing
    dummy_input_1d = torch.randn(2, 1, 512)
    dummy_input_2d = torch.randn(2, 1, 64, 64)

    config_files = [f for f in os.listdir(CONFIG_FOLDER) if f.endswith(".yaml") or f.endswith(".yml")]
    total = len(config_files)

    for idx, fname in enumerate(config_files, start=1):
        path = os.path.join(CONFIG_FOLDER, fname)
        
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        print(f"\n[INFO] Testing config [{idx}/{total}]: {fname}")
        
        try:
            # Select dummy input shape based on encoder type
            dummy_input = dummy_input_2d if config["encoder"]["type"] == "2D" else dummy_input_1d
            encoder, kan_head = init_modules(config, dummy_input)
            out = kan_head(dummy_input, encoder=encoder)
            print(f"[SUCCESS] Output shape: {out.shape}")
        except Exception as e:
            raise ValueError(f"[ERROR] Failed on config '{fname}': {e}")
        
    # ADD PROFILING with torchsummary and store on log file
