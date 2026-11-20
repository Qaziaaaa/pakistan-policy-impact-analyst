import time
from typing import Dict, Any

# Gemini Flash pricing as of current tier:
# $0.075 per 1,000,000 input tokens
# $0.30  per 1,000,000 output tokens
COST_PER_MILLION_INPUT = 0.075
COST_PER_MILLION_OUTPUT = 0.30

class ObservabilityTracker:
    """
    Production telemetry tracker for multi-agent workflows.
    Records per-agent execution latency, input/output tokens, and dollar cost.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.node_latencies: Dict[str, float] = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self._step_start = time.time()

    def start_step(self):
        self._step_start = time.time()

    def end_step(self, node_name: str, input_text: str = "", output_text: str = ""):
        elapsed = round(time.time() - self._step_start, 3)
        self.node_latencies[node_name] = elapsed

        # Standard approximation: ~4 characters per token
        in_tokens = len(input_text) // 4 if input_text else 0
        out_tokens = len(output_text) // 4 if output_text else 0

        self.total_input_tokens += in_tokens
        self.total_output_tokens += out_tokens

    def get_metrics(self) -> Dict[str, Any]:
        total_latency = round(time.time() - self.start_time, 2)
        total_tokens = self.total_input_tokens + self.total_output_tokens

        # Calculate exact estimated cost
        input_cost = (self.total_input_tokens / 1_000_000) * COST_PER_MILLION_INPUT
        output_cost = (self.total_output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT
        total_cost = round(input_cost + output_cost, 6)

        return {
            "total_latency_seconds": total_latency,
            "total_tokens": total_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": f"${total_cost:.6f}",
            "node_latencies": self.node_latencies
        }
