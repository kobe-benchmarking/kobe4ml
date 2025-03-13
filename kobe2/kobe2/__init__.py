from .operator import run as operate
# from .evaluator import run as evaluate

def kobe(input, output):
    """
    Run the operator first, then the evaluator.
    """
    operate(input, output)
    # evaluate(input, output)

__all__ = ["kobe"]