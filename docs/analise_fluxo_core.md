# Análise do fluxo do core (vetorização)

## Diagnóstico

Foi encontrada uma inconsistência de forma no fluxo vetorizado:

- `NeuronioDendritico.forward` retornava escalar quando `batch_size == 1`.
- `Camada.forward` empilha as saídas dos neurônios esperando tensores com dimensão de batch (`[batch]`).
- Com batch unitário, o `stack` resultava em formato `[n_neurons]` em vez de `[1, n_neurons]`, quebrando a consistência do pipeline entre camadas/projeções.

A correção foi manter o retorno sempre no formato vetorizado `[batch]`, inclusive quando `batch=1`.

## Diagrama de fluxo (tensor shapes)

```mermaid
flowchart LR
    A[Entrada\n[batch, in_features]] --> B[ConexaoDensa\n'bi,oids->bods']
    B --> C[Tensor dendrítico\n[batch, n_neurons, D, S]]
    C --> D[Camada.forward\nloop neurônios]
    D --> E[NeuronioDendritico.forward\n[batch, D, S] -> [batch]]
    E --> F[Stack por neurônio\n[batch, n_neurons]]
    F --> G[Próxima projeção/camada]
```

## Checagem de consistência

- Antes da correção:
  - batch > 1: formato correto (`[batch, n_neurons]`)
  - batch = 1: formato inconsistente (`[n_neurons]`)
- Depois da correção:
  - batch > 1 e batch = 1: formato unificado (`[batch, n_neurons]`)

## Conclusão

O fluxo de vetorização fica coerente após o ajuste: as fronteiras entre `Conexao* -> Camada -> Conexao*` preservam a dimensão de batch de forma estável em todos os casos.
