import torch
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create logger
logger = logging.getLogger(__name__)


class CausalAttention(torch.nn.Module):
    def __init__(self, _inputs: torch.tensor, debug: bool = False):
        super().__init__()
        self.inputs = _inputs
        self.debug = debug
        d_in = _inputs.shape[1]
        d_out = 2
        torch.manual_seed(42)
        self.w_q = torch.nn.Linear(d_in, d_out, bias=False)
        self.w_k = torch.nn.Linear(d_in, d_out, bias=False)
        self.w_v = torch.nn.Linear(d_in, d_out, bias=False)
        if debug:
            logger.info(f"query weights: {self.w_q}")
            logger.info(f"key weights: {self.w_k}")
            logger.info(f"value weights: {self.w_v}")
        self.q = None
        self.k = None
        self.v = None
        self.attention_scores = None
        self.attention_weights = None

    def forward(self) -> torch.tensor:
        self._compute_qkv(self.inputs)
        self._compute_attention_scores()
        self._compute_attention_weights()
        context_vectors = self._compute_context_vectors()
        return context_vectors

    def _compute_qkv(self, _inputs):
        self.q = self.w_q(_inputs)
        self.k = self.w_k(_inputs)
        self.v = self.w_v(_inputs)
        if self.debug:
            logger.info(f"query: {self.q}")
            logger.info(f"key: {self.k}")
            logger.info(f"value: {self.v}")

    def _compute_attention_scores(self):
        self.attention_scores = self.q @ self.k.T
        if self.debug:
            logger.info(f'attention_scores shape: {self.attention_scores.shape}')
            logger.info(f'attention_scores: {self.attention_scores}')

    def _compute_attention_weights(self):
        # standard torch implementation of softmax
        temp_attention_scores = self.attention_scores / self.k.shape[-1]**0.5
        self.attention_weights = torch.softmax(temp_attention_scores, dim=-1)
        if self.debug:
            logger.info(f'attention_weights shape: {self.attention_weights.shape}')
            logger.info(f'attention_weights: {self.attention_weights}')

        # Keep only lower triangle and diagonal, zero out upper triangle
        self.attention_weights = self.attention_weights * torch.tril(torch.ones_like(self.attention_weights))

        if self.debug:
            logger.info(f'attention_weights shape: {self.attention_weights.shape}')
            logger.info(f'attention_weights: {self.attention_weights}')


    def _compute_context_vectors(self) -> torch.tensor:
        _context_vectors = self.attention_weights @ self.v
        if self.debug:
            logger.info(f'context_vectors shape: {_context_vectors.shape}')
            logger.info(f'context_vectors: {_context_vectors}')
        return _context_vectors


if __name__ == '__main__':
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    causal_attention = CausalAttention(_inputs=inputs, debug=True)
    cntxt_vectors = causal_attention.forward()
    logger.info(f"Context Vectors: {cntxt_vectors}")