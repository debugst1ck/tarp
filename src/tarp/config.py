class LstmConfig:
    embedding_dimension = 256
    hidden_dimension = 512
    number_of_layers = 3
    dropout = 0.2
    bidirectional = True
    number_of_iterations = 20


class HyenaConfig:
    model_dimension: int = 256
    number_of_layers: int = 4
    number_of_heads: int = 1
    recurrence_depth: int = 2
    mixing_width: int = 2
    local_context_size: int = 3
    dropout: float = 0.1


class Dnabert2Config:
    hidden_dimension = 768
    number_of_iterations = 10


class TransformerConfig:
    embedding_dimension = 256
    feedforward_dimension = 512
    number_of_layers = 3
    number_of_heads = 4
    dropout = 0.2
    number_of_iterations = 30
