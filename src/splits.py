"""Session-level data split — fixed BEFORE any model is trained (prevents leakage and
'peeking' at test results).

development       : used for training, hyper-parameter tuning and grouped cross-validation.
test_temporal     : the chronologically LAST female session of each class. Same person, later
                    recordings -> tests generalization to new sessions over time.
test_cross_person : the male participant's squash sessions (High) and the sleep session with no
                    subject metadata (Low) -> tests generalization to other / unknown people.
Both test sets are used exactly once, with the final locked model (Phase 6).
"""
import pandas as pd

MAIN_SUBJECT = "female|30-40|170cm-180cm"


def assign_split(sessions: pd.DataFrame) -> pd.Series:
    """`sessions` has one row per session: session_id, strain_label, subject, start_ms."""
    split = pd.Series("development", index=sessions.index)
    split[sessions.subject != MAIN_SUBJECT] = "test_cross_person"
    main = sessions[sessions.subject == MAIN_SUBJECT]
    last_per_class = main.sort_values("start_ms").groupby("strain_label").tail(1).index
    split[last_per_class] = "test_temporal"
    return split
