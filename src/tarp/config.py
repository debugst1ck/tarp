class LstmConfig:
    embedding_dimension = 256
    hidden_dimension = 512
    number_of_layers = 3
    dropout = 0.2
    bidirectional = True
    num_of_iterations = 20


class HyenaConfig:
    embedding_dimension = 256
    hidden_dimension = 512
    number_of_layers = 3
    dropout = 0.2
    num_of_iterations = 20


class Dnabert2Config:
    hidden_dimension = 768
    num_of_iterations = 40


class TransformerConfig:
    embedding_dimension = 256
    hidden_dimension = 512
    number_of_layers = 3
    number_of_heads = 4
    dropout = 0.2
    num_of_iterations = 30
