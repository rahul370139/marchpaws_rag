"""Finite‑state machine for the MARCH‑PAWS sequence.

This module defines a simple state machine that walks through the nine phases
of care defined by the MARCH‑PAWS mnemonic.  The machine starts in the first
state (Massive hemorrhage) and advances one state at a time.  Once the final
state (Splints) has been reached, further calls to `advance()` will keep
returning the terminal state "END".
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class MARCHPAWSStateMachine:
    """A finite‑state controller for MARCH‑PAWS."""

    # Ordered list of states
    sequence: List[str] = field(default_factory=lambda: ["M", "A", "R", "C", "H", "P", "A2", "W", "S"])
    state_index: int = 0  # Start at the first state

    @property
    def current_state(self) -> str:
        if self.state_index < len(self.sequence):
            return self.sequence[self.state_index]
        return "END"

    def advance(self) -> str:
        """Move to the next state.  Returns the new state identifier."""
        if self.state_index < len(self.sequence):
            self.state_index += 1
        return self.current_state

    def reset(self) -> None:
        """Reset the state machine to the initial state."""
        self.state_index = 0

    def has_more(self) -> bool:
        """True if the machine is not yet at the terminal state."""
        return self.current_state != "END"
    
    def get_next_state(self) -> str:
        """Get the next state without advancing."""
        if self.state_index + 1 < len(self.sequence):
            return self.sequence[self.state_index + 1]
        return "END"