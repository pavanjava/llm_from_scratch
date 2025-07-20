import torch
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create logger
logger = logging.getLogger(__name__)


class SelfAttention:
    def __init__(self, _inputs: torch.tensor, debug: bool = False):
        self.inputs = _inputs
        d_in = _inputs.shape[1]
        d_out = 2
        torch.manual_seed(42)
        self.w_q = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.w_k = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.w_v = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        if debug:
            logger.info(f"query weights: {self.w_q}")
            logger.info(f"key weights: {self.w_k}")
            logger.info(f"value weights: {self.w_v}")


if __name__ == '__main__':
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],  # Your     (x^1)
         [0.55, 0.87, 0.66],  # journey  (x^2)
         [0.57, 0.85, 0.64],  # starts   (x^3)
         [0.22, 0.58, 0.33],  # with     (x^4)
         [0.77, 0.25, 0.10],  # one      (x^5)
         [0.05, 0.80, 0.55]]  # step     (x^6)
    )
    self_attention = SelfAttention(_inputs=inputs, debug=True)

