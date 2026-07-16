"""
A simplified architecture of the Temporal Fusion Transformer (TFT) designed by
Lim et al. in 2019. The full architecture model can be found in Figure 2 that
is on page 6 of the paper.

The components that are included are:
    * Variable Selection
    * LSTM Encoder/Decoder
    * Dropout, Gate, Add & Norm
    * GRN (Gated Residual Network)
    * Masked Interpretable Multi-head Attention
    * Dense (Linear layer)

Components that are ommitted:
    * Static Covariate Encoders
    * Quantile Forecasts
"""
import torch
import torch.nn as nn

class TemporalFusionTransformer(nn.Module):
    """
    Leverages all architecture blocks and functions constructed below to build the overarching model.
    """
    def __init__(self, past_feature_names, future_feature_names, input_size=64,
                 hidden_size=64, n_heads=1, dropout=0.1, encoder_len=168, decoder_len=24):
        super().__init__()
        # *** Ingredients (architecture components) for the entire model ***
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len

        # Project each raw scalar feature (dim 1) up to input_size before VSN takes it in.
        # One Linear/Fully Connected Layer per feature name, covering both encoder and decoder feature sets.
        all_feature_names = set(past_feature_names) | set(future_feature_names)
        self.feature_embeddings = nn.ModuleDict({
            name: nn.Linear(1, input_size) for name in all_feature_names
        })

        self.encoder_vsn = VariableSelectionNetwork(past_feature_names, input_size, hidden_size, dropout)
        self.decoder_vsn = VariableSelectionNetwork(future_feature_names, input_size, hidden_size, dropout)

        self.lstm = LSTMEncoderDecoder(hidden_size, num_layers=1, dropout=0.0)
        self.post_lstm_gate = GateAddNorm(hidden_size, hidden_size, dropout)

        # Static enrichment GRN, context=None in v1, which behaves as a plain 2-layer residual block
        self.static_enrichment = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        self.attention = InterpretableMultiHeadAttention(hidden_size, n_heads, dropout)
        self.post_attn_gate = GateAddNorm(hidden_size, hidden_size, dropout)

        self.position_wide_ff = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.post_ff_gate = GateAddNorm(hidden_size, hidden_size, dropout)

        self.output_layer = nn.Linear(hidden_size, 1) # single-point forecast
    
    def embed_(self, feature_dict):
        # feature_dict[name]: (B, T, 1) raw --> (B, T, input_size) embedded
        return {name: self.feature_embeddings[name](feature_dict[name]) for name in feature_dict}

    def forward(self, past_feature_dict, future_feature_dict):
        past_feature_dict = self.embed_(past_feature_dict)
        future_feature_dict = self.embed_(future_feature_dict)

        # Variable Selection
        encoder_vsn_out, encoder_weights = self.encoder_vsn(past_feature_dict)   # (B, 168, H)
        decoder_vsn_out, decoder_weights = self.decoder_vsn(future_feature_dict) # (B, 24, H)

        # LSTM Encoder/Decoder
        lstm_out = self.lstm(encoder_vsn_out, decoder_vsn_out)                   # (B, 192, H)
        vsn_residual = torch.cat([encoder_vsn_out, decoder_vsn_out], dim=1)      # (B, 192, H)

        # Dropout --> Gate --> Add & Norm (post-LSTM)
        gated = self.post_lstm_gate(lstm_out, vsn_residual)                      # (B, 192, H)

        # GRN (static enrichment where context=None)
        enriched = self.static_enrichment(gated)                                 # (B, 192, H)

        # Masked Interpretable Multi-Head Attention
        # Queries: only need the 24 decoder positions need forecasts.
        # Keys/values: the full 192 (168 past and 24 future) so each forecast hour can look anywhere causally valid.
        q = enriched[:, -self.decoder_len:, :]                                   # (B, 24, H)
        k = v = enriched                                                         # (B, 192, H)

        mask = causal_mask(self.decoder_len, self.encoder_len + self.decoder_len, enriched.device)
        attn_out, attn_weights = self.attention(q, k, v, mask=mask)              # (B, 24, H), (B, 24, 192)

        # Droput --> Gate --> Add & Norm (post-attention)
        attn_gated = self.post_attn_gate(attn_out, q)

        # GRN (position-wise feed-forward)
        ff_out = self.position_wide_ff(attn_gated)                               # (B, 24, H)

        # Gate --> Add & Norm (final)
        final = self.post_ff_gate(ff_out, attn_gated)                            # (B, 24, H)

        # Dense output
        forecast = self.output_layer(final).squeeze(-1)                          # (B, 24)

        return forecast, {
            "encoder_vsn_weights": encoder_weights,     # (B, 168, num_past_features)
            "decoder_vsn_weights": decoder_weights,     # (B, 24, num_future_features)
            "attention_weights": attn_weights,          # (B, 24, 192)
        }

def causal_mask(decoder_len, total_len, device):
    # Ensures that the decoder position t cannot attend to positions past t
    encoder_len = total_len - decoder_len
    mask = torch.ones(decoder_len, total_len, dtype=torch.bool, device=device)
    for t in range(decoder_len):
        mask[t, :encoder_len + t + 1] = False
    return mask

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN)
    Structure: a (+ external context (optional)) --> Dense (Linear) --> ELU --> Dense (Linear)
                --> Dropout --> GLU gate --> Add & LayerNorm (+ residual)

    Arguments:
        input_size: dimension of primary input a
        hidden_size: internaal GRN width
        output_size: output dim (residual is projected to match if input_size != output_size)
        dropout: dropout applied before the gate
        context_size: dim of optional external context c (None if this GRN never receives context)

    Purpose: employs gates and residual connections to optimize data flow
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, context_size=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Optional context projection inserted into first dense (linear) layer
        self.context_proj = nn.Linear(context_size, hidden_size, bias=False) if context_size else None

        self.fc1 = nn.Linear(input_size, hidden_size) # eta_2 = ELU(W2 a + W3 c + b2)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size) # eta_1 = W1 eta_2 + b1
        self.dropout = nn.Dropout(dropout)

        # Gated Linear Unit (GLU): sigmoid(W4 eta_1 + b4) * (W5 eta_1 + b5), output size = output_size
        self.gate = nn.Linear(hidden_size, output_size * 2)

        # If the dimensions differ, project the residual connection to the output size
        self.skip_proj = nn.Linear(input_size, output_size) if input_size != output_size else None
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, a, c=None):
        """
        a (primary input): (..., input_size)
        c (optional external context): (..., context_size)
        returns: (..., output_size)
        """
        residual = a if self.skip_proj is None else self.skip_proj(a)

        x = self.fc1(a)
        if self.context_proj is not None and c is not None:
            x = x + self.context_proj(c)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        gate_a, gate_b = self.gate(x).chunk(2, dim=-1)  # split into 2 halves
        gated = torch.sigmoid(gate_a) * gate_b          # layer based on GLU

        return self.layer_norm(gated + residual)        # Add & Norm

  
class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN)
    Multiple GRNs operated on each transformed input variable indepently and a single GRN takes in 
    flattened inputs and external context (optional) that produce softmax variable selection weights.
    The outputs are combined as a weighted sum. Variable selection weights are the TFT's per-timestep
    feature importance (FIRST interpretability output).

    Arguments:
        feature_names: ordered list of feature names, which fixes processing order and count
        input_size: dimension of each varible after being linearly projected (d_model)
        hidden_size: GRN width and also the VSN output dimension
        dropout: dropout for internal GRNs
        context_size: dimension of optional static external context (None if unused)
    
    Purpose: select the most actively used input feature.
    """
    def __init__(self, feature_names, input_size, hidden_size, dropout=0.1, context_size=None):
        super().__init__()
        self.feature_names = feature_names
        self.num_features = len(feature_names)

        # One GRN per input: (B, T, input_size) --> (B, T, hidden_size)
        self.feature_grns = nn.ModuleDict({
            name: GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for name in feature_names
        })

        # GRN over the flattened concatentation of all input --> per-feature selection logits
        self.flattened_grn = GatedResidualNetwork(
            input_size=self.num_features * input_size,
            hidden_size=hidden_size,
            output_size=self.num_features,
            dropout=dropout,
            context_size=context_size,
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, feature_dict, context=None):
        """
        feature_dict: dict[name -> (B, T, input_size)], one tensor per name in self.feature_names
        context: (B, context_size) optional static context

        returns:
            combined: (B, T, hidden_size)    * the VSN's weighted-combined output
            weights: (B, T, num_features)    * interpretability output: importance of each feature at each timestep
        """
        # Per-input GRNs, stacked: (B, T, num_features, hidden_size)
        transformed = torch.stack(
            [self.feature_grns[name](feature_dict[name]) for name in self.feature_names],
            dim=2
        )

        # Concatenate raw (pre-GRN) input for the selection GRN: (B, T, num_features * input_size)
        flattened = torch.cat([feature_dict[name] for name in self.feature_names], dim=-1)

        # Broadcast static context over the time dimension if provided
        if context is not None and context.dim() == 2:
            context = context.unsqueeze(1).expand(-1, flattened.size(1), -1)
        
        # Selection weights: (B, T, num_features) and softmax over feature axis
        weights = self.softmax(self.flattened_grn(flattened, context))

        # Weighted sum over features --> (B, T, hidden_size)
        combined = (weights.unsqueeze(-1) * transformed).sum(dim=2)

        return combined, weights


class LSTMEncoderDecoder(nn.Module):
    """
    Sequence-to-sequence LSTM block
    This block comprises two separate LSTM modules, which do not share weights.
    Therefore, the encoder learns dynamics from the past and the decoder learns dynamics
    over the future window. The encoder's final (hidden, cell) state seeds the decoder's initial state.
    This is the point where information from the past reaches the forecast window.
    The initial states equal None given the absence of the static covariate encoders. 

    Purpose: process short-term temporal relationships from past and known future inputs.
    """
    def __init__(self, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder: consumes VSN output over the past window, e.g., (B, 168, hidden_size)
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers= num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, 
        )

        # Decoder: consumes VSN output over the known future window, e.g., (B, 24, hidden_size)
        # Same shape convention as encoder but with a distinct set of weights.
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
    
    def forward(self, encoder_input, decoder_input, init_hidden=None, init_cell=None):
        """
        encoder_input: (B, 168, hidden_size) / encoder VSN combined output
        decoder_input: (B, 24, hidden_size) / decoder VSN combined output
        init_hidden, init_cell: (num_layers, B, hidden_size) {None since static covariate encoders are absent}
        
        returns:
            lstm_output: (B, 164+28, hidden_size) 
                encoder and decoder outputs concatenated along the time axis to be subsequently
                passed into Gate & Add Norm after dropiout alongside the pre-LSTM VSN outputs as the residual.
        """
        B = encoder_input.size(0)
        device = encoder_input.device

        if init_hidden is None:
            init_hidden = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
        if init_cell is None:
            init_cell = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
        
        # Run encoder over the past window and keep its final (hidden, cell) state
        encoder_output, (enc_hidden, enc_cell) = self.encoder_lstm(
            encoder_input, (init_hidden, init_cell)
        )

        # Decoder is seeded with the encoder's final state which is how information from
        # the past 168 hours influences the 24-hour forcast window and kept separate from
        # whatever the decoder's known future inputs provide.
        decoder_output, _ = self.decoder_lstm(
            decoder_input, (enc_hidden, enc_cell)
        )

        # Concatenate along the time axis: (B, 168, hidden_size) + (B, 24, hidden_size) = (B, 192, hidden_size)
        lstm_output = torch.cat([encoder_output, decoder_output], dim=1)
        return lstm_output


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention
    Standard scaled dot-product attention per head, but V is a single shared projection
    across all heads unlinke one V per head like vanilla transformer attention. This is what
    enables the averaging of the per-head attention weights into one interpretable weight matrix, which
    is the SECOND interpretability output, showing which lookback positions matter for each forecast hour.

    Arguments:
        hidden_size: model dimension (d_model) must be divisible by n_heads
        n_heads: number of attention heads
        dropout: dropout applied to attention weights

    Purpose: capture long-term temporal relationships from inputs.
    """
    def __init__(self, hidden_size, n_heads=1, dropout=0.1):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        # Separate Q, K projection per head --> one big Linear, split into heads after
        self.q_proj = nn.Linear(hidden_size, hidden_size) # reshaped into (n_heads, head_dim)
        self.k_proj = nn.Linear(hidden_size, hidden_size)

        # Shared value projection --> one V among all heads, which is the paper's
        # specific approach that makes head-averaging powerful.
        self.v_proj = nn.Linear(hidden_size, self.head_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.head_dim, hidden_size) # W_H in the paper

    def forward(self, q, k, v, mask=None):
        """
        q: (B, Tq, hidden_size)
            ---> queries, e.g. the 24 decoder positions
        k: (B, Tk, hidden_size)
            --> keys, e.g. all 168+24 encoder+decoder positions
        v: (B, Tk, hidden_size)
            --> values with same positions as k (projected down to head_dim and shared)
        mask: (Tq, Tk) bool or float, True/large-negative where attention is not allowed
            (causal mask --> this means query at decoder position t cannot attend to decoder positions > t)
        
        returns:
            output: (B, Tq, hidden_size)    * attended output with same dim as input for residual add
            attn_weights: (B, Tq, Tk)       * averaged across heads: interpretability output #2
        """
        B, Tq, _ = q.shape
        Tk = k.size(1)

        # Project and split Q, K into heads: (B, T, hidden_size) -> (B, n_heads, T, head_dim)
        Q = self.q_proj(q).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.q_proj(k).view(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        # Shared V: projected once, not split by head -> (B, Tk, head_dim)
        V = self.v_proj(v)

        # Scaled dot-product scores per head: (B, n_heads, Tq, Tk)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        
        attn = self.dropout(torch.softmax(scores, dim=-1)) # (B, n_heads, Tq, Tk)

        # Apply each head's attention weights to the same shared V and then average heads.
        # Since V is shared, averaging the outputs which is the same as averaging the
        # attention weights themselves
        head_outputs = torch.matmul(attn, V.unsqueeze(1)) # (B, n_heads, Tq, head_dim)
        combined = head_outputs.mean(dim=1)               # (B, Tq, head_dim)

        output = self.out_proj(combined)    # (B, Tq, hidden_size)
        attn_weights = attn.mean(dim=1)     # (B, Tq, Tk), averaged over heads

        return output, attn_weights


class GLU(nn.Module):
    """
    Gated Linear Unit: sigmoid(W_a x) * (W_b x) where the two halves are computed in one fused Linear.
                       Uses some of the architecture and data transformations found in GRN.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size * 2)
    
    def forward(self, x):
        a, b = self.linear(x).chunk(2, dim=-1)
        return torch.sigmoid(a) * b


class GateAddNorm(nn.Module):
    """Dropout --> GLU gate --> residual add --> Layer Norm. The Gate + Add & Norm pair
    that appears after the LSTM Encoder/Decoder block and after the temporal
    self-attention. This module also contains certain elements found in GRN."""
    def __init__(self, input_size, output_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(input_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, residual):
        x = self.dropout(x)
        x = self.glu(x)
        return self.layer_norm(x + residual)
