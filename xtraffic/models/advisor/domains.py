"""Phase 19 — DOMAIN PROFILES: the vocabulary the advisor speaks in.

WHY THIS FILE EXISTS
--------------------
Phase 19 runs the SAME pipeline (ST-GNN -> GNNExplainer -> LLM advisor) on the
IEEE 14-bus power grid. Everything numeric ports for free — that is the whole
cross-domain claim, and `DIFFERENCES_POWER_GRID.md` §1 shows the tensor contract
is byte-identical. But one thing does NOT port: the WORDS.

The Phase-4 prompt hard-codes traffic vocabulary. Pointed at a power grid it says:

    You are a traffic-operations advisor.
      Current speed: 0.88 mph
      Predicted speed in 30 min: 0.95 mph
    TOP CONTRIBUTING SENSORS ...
    1) Explain why this congestion is predicted ...

Every number there is CORRECT (0.88 really is the bus voltage) and every label is
a LIE. `DIFFERENCES_POWER_GRID.md` §8.2 predicted exactly this and flagged it as
owed: "numbers will be right, LABELS WILL LIE at ~0.005 pu."

For a plot axis a wrong label is cosmetic. For an LLM prompt it is NOT: the model
reasons about the semantics of what it is told. Telling llama3.1 that a bus is
travelling at 0.88 mph and asking for traffic interventions invites it to invent
ramp metering for a substation. We would then measure "hallucination" that our own
prompt caused — a confound that would silently destroy the cross-domain result.
So the vocabulary is parameterised here, and ONLY the vocabulary.

WHAT IS AND IS NOT A DOMAIN DIFFERENCE (this distinction is the contribution)
---------------------------------------------------------------------------
Ported unchanged, zero edits: the model, the explainer, the faithfulness metric,
the tensor contract, the advisory JSON contract, the prompt STRUCTURE
(SYSTEM -> CONTEXT -> EXPLANATION -> TASK), and the grounding instruction
("cite ONLY what is in the explanation"). Those carry the scientific claim.

Swapped here: unit strings, node nouns, the name of the stress phenomenon, and
the operator role. Those are labels, not mechanism.

DEFAULT-PRESERVING (CLAUDE.md: never silently change something we built earlier)
-------------------------------------------------------------------------------
TRAFFIC is the default everywhere. Its strings are copied VERBATIM from the
Phase-4 advisor.py, so every existing METR-LA / PEMS-BAY / Chicago caller
produces a BYTE-IDENTICAL prompt. `evaluation/verify_traffic_unchanged.py`
asserts this against prompts captured before the refactor, for the committed
Phase 4/5/11/15b/16/17/18 numbers.

Python 3.9 compatible.
"""
from __future__ import annotations

from typing import Dict, Optional


class DomainProfile:
    """The words one domain uses for the quantities the pipeline computes.

    Deliberately dumb: strings and a formatter, no behaviour. Anything with logic
    in it would be a second place for the pipeline to differ across domains, and
    the entire point of Phase 19 is that the pipeline does NOT differ.
    """

    def __init__(self, key: str, role: str, unit: str, quantity: str,
                 node_noun: str, stress: str, network: str,
                 invent_list: str, value_decimals: Optional[int],
                 free_state: str, stress_direction: str):
        self.key = key
        self.role = role                    # "traffic-operations advisor"
        self.unit = unit                    # "mph" | "pu"
        self.quantity = quantity            # "speed" | "voltage"
        self.node_noun = node_noun          # "sensor" | "bus"
        self.stress = stress                # "congestion" | "undervoltage"
        self.network = network              # "city infrastructure" | "grid infrastructure"
        # The things the SYSTEM prompt forbids the model from inventing. Domain
        # specific because "do not invent roads" means nothing to a grid operator.
        self.invent_list = invent_list
        self.value_decimals = value_decimals
        self.free_state = free_state        # "free-flowing" | "at nominal voltage"
        self.stress_direction = stress_direction  # what "worse" means, in words

    # Plural forms, derived so there is one source of truth for the noun.
    @property
    def node_plural(self) -> str:
        return self.node_noun + ("es" if self.node_noun.endswith("s") else "s")

    @property
    def node_plural_upper(self) -> str:
        return self.node_plural.upper()

    def fmt(self, value) -> str:
        """Render a quantity with its unit: '54.4 mph' or '0.881 pu'.

        `value_decimals=None` means "print the number exactly as stored", which is
        what Phase 4 did (`"{} mph".format(v)`). TRAFFIC uses None so its rendered
        bytes are unchanged — rounding to 2dp would turn "54.4 mph" into
        "54.40 mph" and silently alter every committed traffic prompt.

        POWER_GRID uses 3 because voltages live in a ~0.3 pu band: at 2dp two buses
        differing by a meaningful few mV both print "0.88", hiding the very
        differences the explainer ranked them on.
        """
        if self.value_decimals is None:
            return "{} {}".format(value, self.unit)
        try:
            return "{:.{d}f} {u}".format(float(value), d=self.value_decimals,
                                         u=self.unit)
        except (TypeError, ValueError):
            return "{} {}".format(value, self.unit)


# --- TRAFFIC: strings copied verbatim from Phase-4 advisor.py. Do not edit ---
# without re-running evaluation/verify_traffic_unchanged.py, which pins them
# against the committed Phase 4/5/11/15b/16/17/18 prompt bytes.
TRAFFIC = DomainProfile(
    key="traffic",
    role="traffic-operations advisor",
    unit="mph",
    quantity="speed",
    node_noun="sensor",
    stress="congestion",
    network="city infrastructure",
    invent_list="sensors, roads, incidents",
    value_decimals=None,           # print as-stored: preserves Phase-4 bytes
    free_state="free-flowing",
    stress_direction="lower speed = worse",
)

# --- POWER GRID (Phase 19) --------------------------------------------------
# Mapping straight out of DIFFERENCES_POWER_GRID.md §2, so the words the advisor
# uses are the ones that file already committed us to:
#   sensor -> bus, speed -> voltage magnitude (pu), congestion -> undervoltage.
# "Voltage sag" rather than "congestion" because that is what a grid operator
# calls it, and the KB (kb/power_grid.json) is written in that vocabulary — the
# advisor and its knowledge base must speak the same language or retrieval and
# reasoning drift apart.
POWER_GRID = DomainProfile(
    key="power_grid",
    role="power-system operations advisor",
    unit="pu",
    quantity="voltage",
    node_noun="bus",
    stress="undervoltage",
    network="grid infrastructure",
    invent_list="buses, lines, transformers, outages",
    value_decimals=3,
    free_state="at nominal voltage",
    stress_direction="lower voltage = worse; 1.00 pu is nominal and "
                     "below 0.95 pu is an undervoltage violation",
)

# City/dataset key -> profile. Anything absent falls back to TRAFFIC, so an
# unknown dataset behaves exactly as it did before this file existed.
_BY_CITY: Dict[str, DomainProfile] = {
    "metr_la": TRAFFIC,
    "pems_bay": TRAFFIC,
    "chicago": TRAFFIC,
    "power_grid": POWER_GRID,
}


def domain_for(city: Optional[str]) -> DomainProfile:
    """Profile for a city/dataset key. Unknown or None -> TRAFFIC (the default
    that preserves every pre-Phase-19 behaviour)."""
    if not city:
        return TRAFFIC
    return _BY_CITY.get(str(city).lower(), TRAFFIC)


def domain_for_explanation(exp: Dict) -> DomainProfile:
    """Profile implied by an explanation dict, read from meta.city (which
    ExplanationBuilder sets to the dataset name). Lets a renderer stay
    domain-correct without every caller threading the domain through by hand."""
    try:
        return domain_for(exp.get("meta", {}).get("city"))
    except AttributeError:
        return TRAFFIC
