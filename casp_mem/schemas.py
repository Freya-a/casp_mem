from dataclasses import dataclass, field

@dataclass(frozen=True)
class Span:
    id: int
    start: int
    end: int
    kind: str = 'ASSERTION'
    scope: str = 'NONE'
    modifies_id: int | None = None
    preconditions: tuple[str, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class ScopeDecision:
    span_id: int
    label: str
    reason: str

@dataclass(frozen=True)
class PatchOp:
    span_id: int
    operation: str
    text: str
    anchors: tuple[str, ...] = field(default_factory=tuple)
    old_branch: str | None = None
    new_branch: str | None = None
    op_mode: str | None = None
