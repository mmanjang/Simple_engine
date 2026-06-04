"""
agent.py
────────
Agent dataclass. Parses a DYCONET cognitive passport and exposes
value weights, beliefs, and mode availability helpers.

Passport format expected (profile.needs keys):
  pro_env, physical, privacy, autonomy, cost, speed,
  safety_accident, safety_crime, comfort, reliable, health_infection
"""

from dataclasses import dataclass, field
from typing import Optional
from value_model import VALUE_DIMENSIONS, MODE_BELIEF_REQUIREMENTS


@dataclass
class Agent:
    id: str
    value_weights: dict        # {dimension: float 0–1}
    beliefs: dict              # {owns_car, owns_bike, has_pt_access: bool}
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    #  Constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict, normalise: bool = True) -> "Agent":
        """
        Parse a cognitive passport dict into an Agent.

        Accepts two formats:
          1. Full passport with top-level "cognitive_passport" key
             (as exported by DYCONET)
          2. Legacy flat format with "values" and "beliefs" keys

        In format 1, needs are read from profile.needs and beliefs are
        read from an explicit beliefs block (preferred) or inferred from
        routing_parameters.mode_weights (fallback: weight > 0.01 = can use).
        """

        # Unwrap top-level "cognitive_passport" envelope if present
        if "cognitive_passport" in data:
            data = data["cognitive_passport"]

        # ── Agent ID ──────────────────────────────────────────────────
        agent_id = data.get("agent_id") or data.get("id") or "unknown_agent"

        # ── Need weights ──────────────────────────────────────────────
        # New format: data.profile.needs
        # Legacy format: data.values
        profile    = data.get("profile", {})
        raw_needs  = profile.get("needs") or data.get("values") or {}

        # Fill all 11 dimensions; missing ones default to 0.0
        filled = {dim: float(raw_needs.get(dim, 0.0)) for dim in VALUE_DIMENSIONS}

        if normalise:
            filled = cls._normalise(filled)

        # ── Beliefs ───────────────────────────────────────────────────
        # Priority 1: explicit beliefs block inside the passport
        explicit_beliefs = data.get("beliefs", {})

        # Priority 2: infer from routing_parameters.mode_weights
        routing_params = data.get("routing_parameters", {})
        mode_weights   = routing_params.get("mode_weights", {})
        inferred = {
            "owns_car":      mode_weights.get("car",  0.0) > 0.01,
            "owns_bike":     mode_weights.get("bike", 0.0) > 0.01,
            "has_pt_access": mode_weights.get("pt",   0.0) > 0.01,
        }

        # Explicit beliefs win over inferred ones
        inferred.update({k: bool(v) for k, v in explicit_beliefs.items()})
        final_beliefs = inferred

        # ── Metadata (everything else) ────────────────────────────────
        skip = {"agent_id", "id", "profile", "beliefs", "routing_parameters"}
        metadata = {k: v for k, v in data.items() if k not in skip}

        return cls(
            id            = agent_id,
            value_weights = filled,
            beliefs       = final_beliefs,
            metadata      = metadata,
        )

    # ------------------------------------------------------------------
    #  Normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(values: dict) -> dict:
        """
        Min-max scale raw need scores to [0, 1].
        If all values are equal (span == 0) return them as-is
        so we don't lose the signal entirely.
        """
        vals  = list(values.values())
        v_min = min(vals)
        v_max = max(vals)
        span  = v_max - v_min

        if span == 0:
            # All needs equally weighted — preserve raw value (already 0–1)
            return dict(values)

        return {k: (v - v_min) / span for k, v in values.items()}

    # ------------------------------------------------------------------
    #  Belief helpers
    # ------------------------------------------------------------------

    def available_modes(self) -> list[str]:
        """Return all modes the agent is able to use given their beliefs."""
        return [
            mode for mode, required in MODE_BELIEF_REQUIREMENTS.items()
            if all(self.beliefs.get(b, False) for b in required)
        ]

    def can_use(self, mode: str) -> bool:
        required = MODE_BELIEF_REQUIREMENTS.get(mode, [])
        return all(self.beliefs.get(b, False) for b in required)

    # ------------------------------------------------------------------
    #  Profile type inference (used for distance penalty multipliers)
    # ------------------------------------------------------------------

    def infer_profile_type(self) -> str:
        """
        Map the agent's dominant needs to a profile type used by
        distance penalty curves: biospheric, altruistic, hedonic, egoistic.
        """
        weights = self.value_weights
        sorted_vals = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top1       = sorted_vals[0][0]
        top2       = sorted_vals[1][0] if len(sorted_vals) > 1 else ""
        top1_weight = sorted_vals[0][1]

        # Biospheric: pro_env or physical dominant
        if (top1 in ("pro_env", "physical") and top1_weight > 0.7) or \
           (top1 == "pro_env" and top2 == "physical"):
            return "biospheric"

        # Altruistic: safety needs dominant
        if top1 in ("safety_accident", "safety_crime") and top1_weight > 0.7:
            return "altruistic"

        # Hedonic: comfort dominant
        if top1 == "comfort" and top1_weight > 0.8:
            return "hedonic"

        # Egoistic: autonomy, speed, or privacy dominant
        if top1 in ("autonomy", "speed", "privacy") and top1_weight > 0.8:
            return "egoistic"

        return "egoistic"

    # ------------------------------------------------------------------
    #  Display helpers
    # ------------------------------------------------------------------

    def top_values(self, n: int = 3) -> list[tuple[str, float]]:
        """Return the n most important need dimensions for this agent."""
        return sorted(self.value_weights.items(),
                      key=lambda x: x[1], reverse=True)[:n]

    def summary(self) -> str:
        lines = [f"Agent: {self.id}"]
        lines.append("  Needs (normalised 0–1):")
        for dim, score in sorted(self.value_weights.items(),
                                  key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 10)
            lines.append(f"    {dim:<20} {score:.2f}  {bar}")
        lines.append("  Beliefs:")
        lines.append(f"    owns_car      : {self.beliefs.get('owns_car', False)}")
        lines.append(f"    owns_bike     : {self.beliefs.get('owns_bike', False)}")
        lines.append(f"    has_pt_access : {self.beliefs.get('has_pt_access', False)}")
        lines.append(f"  Available modes: {', '.join(self.available_modes())}")
        return "\n".join(lines)