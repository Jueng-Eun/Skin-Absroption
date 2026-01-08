```mermaid
graph TD

%% Inputs (tensors/objects)
Xp[x_p (B, F_p)]:::obj
Xv[x_v (B, F_v)]:::obj
Xs[x_s (B, F_s)]:::obj
Xe[x_e (B, F_e)]:::obj

%% 4 separate encoder blocks (operations)
subgraph EP[Encoder P]
  direction LR
  P1[Dense proj (F_p, 64)]:::fn --> P2[Self-attn 4h (64, 64)]:::fn --> P3[LayerNorm (64)]:::fn --> P4[FFN ReLU (64, D)]:::fn
end

subgraph EV[Encoder V]
  direction LR
  V1[Dense proj (F_v, 64)]:::fn --> V2[Self-attn 4h (64, 64)]:::fn --> V3[LayerNorm (64)]:::fn --> V4[FFN ReLU (64, D)]:::fn
end

subgraph ES[Encoder S]
  direction LR
  S1[Dense proj (F_s, 64)]:::fn --> S2[Self-attn 4h (64, 64)]:::fn --> S3[LayerNorm (64)]:::fn --> S4[FFN ReLU (64, D)]:::fn
end

subgraph EE[Encoder E]
  direction LR
  E1[Dense proj (F_e, 64)]:::fn --> E2[Self-attn 4h (64, 64)]:::fn --> E3[LayerNorm (64)]:::fn --> E4[FFN ReLU (64, D)]:::fn
end

%% Wire inputs into each encoder
Xp --> P1
Xv --> V1
Xs --> S1
Xe --> E1

%% Encoder outputs (objects)
P4 --> Hp[h_p (B, D)]:::obj
V4 --> Hv[h_v (B, D)]:::obj
S4 --> Hs[h_s (B, D)]:::obj
E4 --> He[h_e (B, D)]:::obj

%% Stack node embeddings
Hp --> COL[Collect 4 (B, D)]:::aux
Hv --> COL
Hs --> COL
He --> COL
COL --> STK[Stack 4 nodes (B, 4, D)]:::fn --> HN[h_nodes (B, 4, D)]:::obj

%% Learnable adjacency (parameter path)
subgraph ADJ[Learnable adjacency A]
  direction LR
  A0[A_logits (4, 4)]:::param --> A1[Sigmoid]:::fn --> A2[Symmetrize]:::fn --> A3[Zero diag]:::fn --> A4[Broadcast (B, 4, 4)]
  A4:::fn
end

%% Message passing + prediction head
HN --> GAT[GAT layer (B, 4, D)]:::fn --> FLT[Flatten (B, 4D)]:::fn --> MLP[MLP head]:::fn --> OUT[y_hat (B, 1)]:::obj
A4 -.-> GAT

%% Styles
classDef obj fill:#ffffff,stroke:#333,stroke-width:1px,color:#000;
classDef fn fill:#e8f0ff,stroke:#2b4a9b,stroke-width:1px,color:#000;
classDef param fill:#fff2df,stroke:#8a5a00,stroke-width:1px,color:#000;
classDef aux fill:#ffffff,stroke:#999,stroke-width:1px,color:#000,stroke-dasharray:4 2;
```
