# Fluxo do core e vetorização

Este diagrama resume o caminho do tensor no núcleo (`Conexao*` → `Camada` → `NeuronioDendritico`) e destaca onde ocorre vetorização.

```mermaid
flowchart TD
    A[Entrada x\n[batch, in_features] ou [batch,1,H,W]] --> B{Tipo de conexão}
    B -->|ConexaoDensa/Esparsa| C[Einsum\nbi,oids->bods]
    B -->|ConexaoRegional| D[Recorte por região\nEinsum bchw,dshw->bds\nStack por neurônio]
    C --> E[Tensor dendrítico\n[batch, out_neurons, D, S]]
    D --> E

    E --> F{Camada.conexao}
    F -->|densa| G[Sem mistura adicional]
    F -->|grid_2d / esparsa| H[Einsum topológico\nbids,oi->bods]
    G --> I[Loop por neurônio]
    H --> I

    I --> J[NeuronioDendritico\nW=log2(1+N)/5]
    J --> K[Potenciais sinápticos\n[batch,D,S] = x * W]
    K --> L[Soma por dendrito\n[batch,D]]
    L --> M[Máscara por theta_dendrito]
    M --> N[Soma total\n[batch]]
    N --> O[Spike surrogate\n(step no forward, gradiente suave no backward)]
    O --> P[Saída da camada\n[batch, out_neurons]]
```

## Checklist rápido de consistência

- **Conexões**: saem em `[batch, out_neurons, D, S]` e alimentam diretamente a `Camada`.
- **Camada**: agora aplica `matriz_conexao` quando `conexao != 'densa'`, preservando topologia configurada.
- **Neurônio**: threshold dendrítico e somático operam com broadcasting correto para batch.
