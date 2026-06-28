#!/usr/bin/env python
import math
import torch
import torch.nn as nn

class ComplexGaborLayer(nn.Module):
    """
    Complex Gabor Wavelet layer for WIRE (Wavelet Implicit Neural Representations).
    Uses complex weights and applies the complex Gabor wavelet activation.
    """
    def __init__(self, in_features, out_features, omega0=10.0, s0=10.0, bias=True, is_first=False, init_type='siren'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega0 = omega0
        self.s0 = s0
        self.is_first = is_first
        self.init_type = init_type
        
        # Real and imaginary components of weight
        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_features))
            self.bias_imag = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.init_type == 'siren':
            if self.is_first:
                # First layer standard uniform U(-1/N, 1/N)
                bound = 1.0 / self.in_features
            else:
                # Hidden layers standard uniform U(-sqrt(6 / (omega0 * N)), sqrt(6 / (omega0 * N)))
                bound = math.sqrt(6.0 / (self.omega0 * self.in_features))
                
            nn.init.uniform_(self.weight_real, -bound, bound)
            nn.init.uniform_(self.weight_imag, -bound, bound)
            if self.bias_real is not None:
                nn.init.uniform_(self.bias_real, -bound, bound)
                nn.init.uniform_(self.bias_imag, -bound, bound)
        elif self.init_type == 'normal':
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.normal_(self.weight_real, std=bound)
            nn.init.normal_(self.weight_imag, std=bound)
            if self.bias_real is not None:
                nn.init.uniform_(self.bias_real, -bound, bound)
                nn.init.uniform_(self.bias_imag, -bound, bound)
        else:  # 'uniform' or default
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.weight_real, -bound, bound)
            nn.init.uniform_(self.weight_imag, -bound, bound)
            if self.bias_real is not None:
                nn.init.uniform_(self.bias_real, -bound, bound)
                nn.init.uniform_(self.bias_imag, -bound, bound)
                
    def forward(self, x):
        # Convert input x to complex if it is real-valued
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
            
        # Complex linear transformation: y = (W_r * x_r - W_i * x_i + b_r) + j * (W_r * x_i + W_i * x_r + b_i)
        real = nn.functional.linear(x.real, self.weight_real) - nn.functional.linear(x.imag, self.weight_imag)
        imag = nn.functional.linear(x.real, self.weight_imag) + nn.functional.linear(x.imag, self.weight_real)
        
        if self.bias_real is not None:
            real = real + self.bias_real
            imag = imag + self.bias_imag
            
        z = torch.complex(real, imag)
        
        # Gabor activation: exp(1j * omega0 * z) * exp(- (s0 * |z|)^2)
        # Using real operations for stability:
        # exponent_real = - omega0 * z_imag - (s0^2) * (z_real^2 + z_imag^2)
        # exponent_imag = omega0 * z_real
        norm_sq = z.real.pow(2) + z.imag.pow(2)
        exp_real = torch.exp(-self.omega0 * z.imag - (self.s0 ** 2) * norm_sq)
        cos_val = torch.cos(self.omega0 * z.real)
        sin_val = torch.sin(self.omega0 * z.real)
        
        return torch.complex(exp_real * cos_val, exp_real * sin_val)

class RealGaborLayer(nn.Module):
    """
    Real Gabor Wavelet layer for WIRE (alternate real-valued formulation).
    Applies standard linear layer followed by sin(omega0 * x) * exp(- (s0 * x)^2).
    """
    def __init__(self, in_features, out_features, omega0=10.0, s0=10.0, bias=True, is_first=False, init_type='siren'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega0 = omega0
        self.s0 = s0
        self.is_first = is_first
        self.init_type = init_type
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.init_type == 'siren':
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / (self.omega0 * self.in_features))
            nn.init.uniform_(self.linear.weight, -bound, bound)
            if self.linear.bias is not None:
                nn.init.uniform_(self.linear.bias, -bound, bound)
        elif self.init_type == 'normal':
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.normal_(self.linear.weight, std=bound)
            if self.linear.bias is not None:
                nn.init.uniform_(self.linear.bias, -bound, bound)
        else:  # uniform
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.linear.weight, -bound, bound)
            if self.linear.bias is not None:
                nn.init.uniform_(self.linear.bias, -bound, bound)
                
    def forward(self, x):
        z = self.linear(x)
        return torch.sin(self.omega0 * z) * torch.exp(- (self.s0 * z).pow(2))

class ComplexLinear(nn.Module):
    """
    Standard Complex-Valued Linear Layer.
    Used as the output layer when complex_weights is True.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_features))
            self.bias_imag = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_real, -bound, bound)
        nn.init.uniform_(self.weight_imag, -bound, bound)
        if self.bias_real is not None:
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)
            
    def forward(self, x):
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
            
        real = nn.functional.linear(x.real, self.weight_real) - nn.functional.linear(x.imag, self.weight_imag)
        imag = nn.functional.linear(x.real, self.weight_imag) + nn.functional.linear(x.imag, self.weight_real)
        
        if self.bias_real is not None:
            real = real + self.bias_real
            imag = imag + self.bias_imag
            
        return torch.complex(real, imag)

class WIRE(nn.Module):
    """
    Wavelet Implicit Neural Representation (WIRE) network.
    Optionally employs complex-valued or real-valued weights, Gabor wavelet activations,
    and supports flexible hyperparameters for configuration.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        hidden_layers: int,
        omega0: float = 10.0,
        s0: float = 10.0,
        omega0_first: float = None,
        s0_first: float = None,
        complex_weights: bool = True,
        bias: bool = True,
        init_type: str = 'siren'
    ):
        """
        Args:
            in_features: Dimension of input (e.g. 2 for coordinates).
            out_features: Dimension of output (e.g. 3 for RGB).
            hidden_features: Dimension of hidden layers (width).
            hidden_layers: Number of hidden layers (depth).
            omega0: Frequency factor of the Gabor wavelet.
            s0: Scaling/spread factor of the Gabor wavelet.
            omega0_first: Frequency factor for the first layer (defaults to omega0).
            s0_first: Scaling/spread factor for the first layer (defaults to s0).
            complex_weights: Use complex weights and activations (main WIRE model).
            bias: Whether to use biases.
            init_type: Initialization strategy ('siren', 'normal', or 'uniform').
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.complex_weights = complex_weights
        
        w_first = omega0_first if omega0_first is not None else omega0
        s_first = s0_first if s0_first is not None else s0
        
        layers = []
        
        # First layer
        if complex_weights:
            layers.append(
                ComplexGaborLayer(
                    in_features,
                    hidden_features,
                    omega0=w_first,
                    s0=s_first,
                    bias=bias,
                    is_first=True,
                    init_type=init_type
                )
            )
        else:
            layers.append(
                RealGaborLayer(
                    in_features,
                    hidden_features,
                    omega0=w_first,
                    s0=s_first,
                    bias=bias,
                    is_first=True,
                    init_type=init_type
                )
            )
            
        # Hidden layers
        for _ in range(hidden_layers - 1):
            if complex_weights:
                layers.append(
                    ComplexGaborLayer(
                        hidden_features,
                        hidden_features,
                        omega0=omega0,
                        s0=s0,
                        bias=bias,
                        is_first=False,
                        init_type=init_type
                    )
                )
            else:
                layers.append(
                    RealGaborLayer(
                        hidden_features,
                        hidden_features,
                        omega0=omega0,
                        s0=s0,
                        bias=bias,
                        is_first=False,
                        init_type=init_type
                    )
                )
                
        self.net = nn.Sequential(*layers)
        
        # Last layer
        if complex_weights:
            self.last_layer = ComplexLinear(hidden_features, out_features, bias=bias)
        else:
            self.last_layer = nn.Linear(hidden_features, out_features, bias=bias)
            
    def forward(self, coords):
        x = coords
        x = self.net(x)
        x = self.last_layer(x)
        
        if self.complex_weights:
            # Return real part as specified in the paper
            return x.real
        else:
            return x

if __name__ == '__main__':
    print("=== Running WIRE module test ===")
    
    # Configuration
    in_features = 2
    out_features = 3
    hidden_features = 64
    hidden_layers = 3
    omega0 = 10.0
    s0 = 10.0
    
    # 1. Complex weights test
    print("\n1. Instantiating Complex-Weights WIRE model:")
    print(f"   Config: in={in_features}, out={out_features}, hidden={hidden_features}, layers={hidden_layers}, omega0={omega0}, s0={s0}, complex_weights=True")
    model_complex = WIRE(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        omega0=omega0,
        s0=s0,
        complex_weights=True
    )
    
    dummy_input = torch.randn(1, 100, in_features)
    print(f"   Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output_complex = model_complex(dummy_input)
    print(f"   Output shape: {output_complex.shape}")
    print(f"   Output dtype: {output_complex.dtype}")
    
    # 2. Real weights test
    print("\n2. Instantiating Real-Weights WIRE model:")
    print(f"   Config: in={in_features}, out={out_features}, hidden={hidden_features}, layers={hidden_layers}, omega0={omega0}, s0={s0}, complex_weights=False")
    model_real = WIRE(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        omega0=omega0,
        s0=s0,
        complex_weights=False
    )
    
    with torch.no_grad():
        output_real = model_real(dummy_input)
    print(f"   Output shape: {output_real.shape}")
    print(f"   Output dtype: {output_real.dtype}")
    
    print("\n=== WIRE module test completed ===")

