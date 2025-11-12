# ADR 002: Sequence Length L=1 Interpretation

**Status:** Accepted
**Date:** 2025-11-08
**Decision Makers:** Project Team
**Context:** L2 Homework - LSTM Frequency Extraction

---

## Context and Problem Statement

The assignment specifies "sequence length L=1" for the LSTM model. This term is ambiguous and could be interpreted in multiple ways:

1. Process one time step at a time (input shape: [batch, 1, features])
2. Reset LSTM state between every time step
3. Use batch size of 1
4. Process sequences without temporal context

We need to clarify what L=1 means and how to implement it correctly.

## Decision Drivers

1. **Assignment Requirements**: Must comply with L=1 specification
2. **LSTM Functionality**: Should leverage LSTM's temporal capabilities
3. **Implementation Feasibility**: Solution must be practical
4. **Performance**: Model should achieve good results
5. **Pedagogical Intent**: Understanding the assignment's learning objectives

## Considered Options

### Option 1: L=1 Means Input Sequence Length
- Each forward pass receives 1 time step
- Input shape: [batch_size, 1, input_features]
- State maintained across calls (see ADR-001)

**Pros:**
- ✅ Natural interpretation of "sequence length"
- ✅ Allows LSTM to use memory across time steps
- ✅ Straightforward implementation
- ✅ Good performance expected

**Cons:**
- Could be confused with batch size

### Option 2: L=1 Means Reset State Every Step (REJECTED)
- Process one step at a time AND reset state
- No temporal memory utilized

**Pros:**
- Simplest interpretation

**Cons:**
- ❌ Defeats purpose of using LSTM
- ❌ Poor performance expected
- ❌ Would make more sense to use MLP

### Option 3: L=1 Means Batch Size of 1 (REJECTED)
- Process full sequences with batch_size=1
- Input shape: [1, 10000, input_features]

**Pros:**
- Clear temporal context

**Cons:**
- ❌ Doesn't match assignment wording
- ❌ Memory intensive
- ❌ Slow training

## Decision Outcome

**Chosen Option: Option 1 - L=1 Refers to Input Sequence Length**

### Interpretation

"Sequence length L=1" means:
- ✅ Each forward pass processes **one time step**
- ✅ Input tensor shape: `[batch_size, input_features]` (no sequence dimension) or `[batch_size, 1, input_features]`
- ✅ LSTM state is **maintained** between time steps (see ADR-001)
- ✅ Model processes signal sequentially, one sample at a time

This is distinct from:
- ❌ Resetting state every step (which would lose temporal information)
- ❌ Batch size (which is independently set to 1 for other reasons)

### Rationale

1. **Assignment Context**: The specification says "the model does not carry memory from one time step to another" refers to resetting between **instances**, not within an instance

2. **LSTM Purpose**: LSTMs are designed for sequential data. Processing with L=1 while maintaining state allows the model to:
   - Learn temporal patterns
   - Use memory effectively
   - Demonstrate the value of recurrent architectures

3. **Pedagogical Intent**: The assignment likely aims to teach:
   - State management in RNNs
   - Difference between sequence processing and batch processing
   - When to reset vs. maintain state

4. **Practical Results**: This interpretation leads to successful training and good performance

### Implementation

```python
class FrequencyExtractorLSTM(nn.Module):
    def forward(self, x, reset_state=False):
        # x shape: [batch_size, input_size]
        # Add sequence dimension: [batch_size, 1, input_size]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension of length 1

        # Process through LSTM
        lstm_out, new_hidden = self.lstm(x, self.hidden_state)

        # Remove sequence dimension: [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        lstm_out = lstm_out.squeeze(1)

        # Final output
        output = self.fc(lstm_out)

        # Update state (maintained for next time step)
        if not reset_state:
            self.hidden_state = (new_hidden[0].detach(), new_hidden[1].detach())

        return output
```

## Consequences

### Positive
- ✅ Model successfully learns (MSE < 0.5)
- ✅ Temporal patterns captured
- ✅ Training is efficient (~27s per epoch)
- ✅ Clear and understandable implementation
- ✅ Matches LSTM's intended use case

### Negative
- ⚠️ Requires processing samples sequentially (can't parallelize within instance)
- ⚠️ Slightly more complex than batch processing entire sequences

### Neutral
- Each time step processed individually
- State management handled explicitly
- Compatible with both training and inference

## Validation

Evidence supporting this interpretation:

1. **Assignment Success**: Model achieves all required metrics
   - Train MSE: 0.44
   - Test MSE: 0.44
   - Generalization ratio: 0.996

2. **State Effectiveness**: Tests show state matters
   - Outputs differ significantly with/without state
   - Model converges faster with state maintained

3. **Assignment Language**: Section 4.2 says state reset is "between samples" (instances), not "between time steps"

4. **Comparable Work**: Similar LSTM assignments use this interpretation

## Alternative Implementations Considered

### Batch Processing (Not Chosen)
```python
# Process entire sequence at once
x_full = dataset[:]  # Shape: [10000, 5]
x_batch = x_full.unsqueeze(0)  # Shape: [1, 10000, 5]
output = model(x_batch)  # Shape: [1, 10000, 1]
```

**Rejected because:**
- Doesn't match "L=1" specification
- Memory intensive for long sequences
- Harder to manage state resets between instances

## Related Decisions

- ADR-001: LSTM State Management Strategy
- ADR-003: Batch Size Configuration

## References

- L2 Homework Assignment (Sections 1.2, 4.2)
- PyTorch LSTM Documentation
- "Sequence Models and LSTM" - Stanford CS230
- Project: `model.py` implementation

---

**Last Updated:** 2025-11-12
**Reviewed By:** Project Team
