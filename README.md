```mermaid
flowchart TD
    Xp[x_p] --> Ep[Encoder_p]
    Xv[x_v] --> Ev[Encoder_v]
    Xs[x_s] --> Es[Encoder_s]
    Xe[x_e] --> Ee[Encoder_e]

  %% Encoders
  subgraph Ep[SelfAttentionEncoder_p]
    Ep1[Dense proj -> D_proj=64] --> Ep2[MHA: heads=4, key_dim=64] --> Ep3[LayerNorm] --> Ep4[FF: Dense(dim_hidden, ReLU)]
  end

  subgraph Ev[SelfAttentionEncoder_v]
    Ev1[Dense proj -> D_proj=64] --> Ev2[MHA: heads=4, key_dim=64] --> Ev3[LayerNorm] --> Ev4[FF: Dense(dim_hidden, ReLU)]
  end

  subgraph Es[SelfAttentionEncoder_s]
    Es1[Dense proj -> D_proj=64] --> Es2[MHA: heads=4, key_dim=64] --> Es3[LayerNorm] --> Es4[FF: Dense(dim_hidden, ReLU)]
  end

  subgraph Ee[SelfAttentionEncoder_e]
    Ee1[Dense proj -> D_proj=64] --> Ee2[MHA: heads=4, key_dim=64] --> Ee3[LayerNorm] --> Ee4[FF: Dense(dim_hidden, ReLU)]
  end

  Ep --> Hp[h_p (B,D)]
  Ev --> Hv[h_v (B,D)]
  Es --> Hs[h_s (B,D)]
  Ee --> He[h_e (B,D)]

  %% Stack nodes
  Hp --> Stack[Stack nodes: h_nodes = [p,v,s,e] -> (B,4,D)]
  Hv --> Stack
  Hs --> Stack
  He --> Stack

  %% Learnable adjacency
  subgraph Adj[Learnable adjacency]
    A0[A_logits (4x4), trainable] --> A1[sigmoid -> A_prob (4x4)]
    A1 --> A2[symmetrize: (A + A^T)/2]
    A2 --> A3[zero diagonal: A*(1-I)]
    A3 --> A4[broadcast -> adj_batch (B,4,4)]
  end

  Stack --> GAT
  Adj --> GAT

  %% GAT + MLP
  subgraph GAT[GraphAttentionLayer]
    G1[Linear: Wh = hW -> (B,4,D)] --> G2[Attention logits e_ij]
    G2 --> G3[Mask with adj_batch]
    G3 --> G4[softmax over j]
    G4 --> G5[Weighted sum -> h_gnn (B,4,D)]
  end

  GAT --> Flat[Flatten: reshape -> (B,4D)]
  Flat --> MLP1[Dropout]
  MLP1 --> MLP2[Dense(ff_dim, ReLU)]
  MLP2 --> MLP3[Dropout]
  MLP3 --> Out[Dense(1) -> y_hat (B,)]
```
