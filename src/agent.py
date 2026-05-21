
from dataclasses import dataclass, field
from typing import Optional
from value_model import VALUE_DIMENSIONS, MODE_BELIEF_REQUIREMENTS


@dataclass
class Agent:
    id: str
    value_weights: dict        
    beliefs: dict              
    metadata: dict = field(default_factory=dict)   

    # Constructors 

    @classmethod
    def from_dict(cls, data: dict, normalise: bool = True) -> "Agent":
  
        agent_id = data.get("id", "unknown_agent")
        raw_values = data.get("values", {})
        beliefs    = data.get("beliefs", {})
        metadata   = {k: v for k, v in data.items()
                      if k not in ("id", "values", "beliefs")}

        # Fill any missing dimensions with 0
        filled = {dim: float(raw_values.get(dim, 0.0))
                  for dim in VALUE_DIMENSIONS}

        if normalise:
            filled = cls._normalise(filled)

        # Default beliefs to False if not provided
        default_beliefs = {
            "owns_car":      False,
            "owns_bike":     False,
            "has_pt_access": False,
        }
        default_beliefs.update({k: bool(v) for k, v in beliefs.items()})

        return cls(
            id            = agent_id,
            value_weights = filled,
            beliefs       = default_beliefs,
            metadata      = metadata,
        )

    @staticmethod
    def _normalise(values: dict) -> dict:
        """
        Scale the raw data if it is not already normalised to 0–1.
        """
        vals  = list(values.values())
        v_min = min(vals)
        v_max = max(vals)
        span  = v_max - v_min

        if span == 0:
            return {k: 0.5 for k in values}

        return {k: (v - v_min) / span for k, v in values.items()}

    #  Belief helpers. Available options for an agents are determined by their beliefs.

    def available_modes(self) -> list[str]:
        
        available = []
        for mode, required_beliefs in MODE_BELIEF_REQUIREMENTS.items():
            if all(self.beliefs.get(b, False) for b in required_beliefs):
                available.append(mode)
        return available

    def can_use(self, mode: str) -> bool:
        required = MODE_BELIEF_REQUIREMENTS.get(mode, [])
        return all(self.beliefs.get(b, False) for b in required)
    
    def infer_profile_type(self) -> str:
       
        weights = self.value_weights
        
        # Get top 2 values
        sorted_values = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top1, top2 = sorted_values[0][0], sorted_values[1][0]
        top1_weight, top2_weight = sorted_values[0][1], sorted_values[1][1]
        
        # Biospheric: pro_environment or physical_activity dominant
        if (top1 in ("pro_environment", "physical_activity") and top1_weight > 0.7) or \
           (top1 == "pro_environment" and top2 == "physical_activity"):
            return "biospheric"
        
        # Altruistic: safety dominant
        if top1 == "safety" and top1_weight > 0.7:
            return "altruistic"
        
        # Hedonic: hedonism or comfort dominant
        if (top1 in ("hedonism", "comfort") and top1_weight > 0.8) or \
           (top1 == "hedonism" and top2 == "comfort"):
            return "hedonic"
        
        # Egoistic: autonomy, speed, or privacy dominant
        if top1 in ("autonomy", "speed", "privacy") and top1_weight > 0.8:
            return "egoistic"
        
        # Default: egoistic (most conservative distance tolerance)
        return "egoistic"

    #  Display helpers

    def top_values(self, n: int = 3) -> list[tuple[str, float]]:
        """Return the n most important value dimensions for this agent."""
        sorted_vals = sorted(self.value_weights.items(),
                             key=lambda x: x[1], reverse=True)
        return sorted_vals[:n]

    def summary(self) -> str:
        lines = [f"Agent: {self.id}"]
        lines.append("  Values (normalised 0–1):")
        for dim, score in sorted(self.value_weights.items(),
                                  key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 10)
            lines.append(f"    {dim:<20} {score:.2f}  {bar}")
        lines.append("  Beliefs:")
        lines.append(f"    owns_car      : {self.beliefs.get('owns_car',False)}")
        lines.append(f"    owns_bike     : {self.beliefs.get('owns_bike',False)}")
        lines.append(f"    has_pt_access : {self.beliefs.get('has_pt_access',False)}")
        lines.append(f"  Available modes: {', '.join(self.available_modes())}")
        return "\n".join(lines)