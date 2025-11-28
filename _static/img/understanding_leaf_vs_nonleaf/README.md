# Diagram Update Required

## Issue #3580: Incorrect requires_grad in comp-graph-1.png

The diagram `comp-graph-1.png` needs to be updated to correct an error:

**Current (Incorrect):**
- Shows `z = x @ W + b` with `requires_grad = False`

**Should be (Correct):**
- Shows `z = x @ W + b` with `requires_grad = True`

**Reason:**
Since both `W` and `b` have `requires_grad = True`, the output tensor `z` must also have `requires_grad = True`. This is a fundamental rule in PyTorch's autograd: when at least one input to an operation has `requires_grad=True`, the output will also have `requires_grad=True`.

This is confirmed in the tutorial code:
```python
W = torch.ones(3, 2, requires_grad=True)  # weights with shape: (3, 2)
b = torch.ones(1, 2, requires_grad=True)  # bias with shape: (1, 2)
z = (x @ W) + b                           # pre-activation with shape: (1, 2)
print(f"{z.requires_grad=}") # prints True because tensor is a non-leaf node
```

**Action Required:**
The diagram image file needs to be regenerated or edited to show `z` with `requires_grad = True` instead of `False`.

A note has been added to the tutorial text to clarify this until the diagram can be updated.
