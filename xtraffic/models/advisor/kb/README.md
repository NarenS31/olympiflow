# City knowledge-base template & schema

Layer 3 (the LLM advisor) is only allowed to cite causes that appear in the
mathematical explanation, but it still needs **city context** to turn "sensor
124 in East LA is slowing" into a concrete, infrastructure-aware recommendation
("meter the I-10 eastbound on-ramps, divert to Olympic Blvd"). That context
lives here, one JSON file per city.

Adding a new city (e.g. Chicago in Phase 6) is a fill-in-the-template job — no
code changes. Register the file in `configs/advisor.yaml` under `cities:`.

## Schema (`{city}.json`)

```json
{
  "city": "<dataset name, matches configs/data.yaml>",
  "display_name": "<human city name>",
  "chunks": [
    {
      "id": "<unique string>",
      "title": "<short human title>",
      "region_tags": ["<lowercase keyword>", "..."],
      "text": "<planner-facing knowledge, factual, publicly sourced>"
    }
  ]
}
```

### Field rules
- **`region_tags`** are the retrieval hooks. They are scored (keyword overlap)
  against the explanation's node names / region labels, so **they MUST use the
  same region strings that `models/explainer/node_names.py` emits** (e.g.
  `"downtown la"`, `"glendale / burbank"`, `"san fernando valley"`). Add
  corridor/freeway keywords too (`"i-10"`, `"i-405"`) so freeway-named causes
  also match.
- **`text`** must be **factual and publicly sourced** (LADOT/Caltrans/Metro for
  LA; CDOT/IDOT/CTA for Chicago). Never invent capacities or signal constraints —
  a reviewer can check them, and the whole point of the layer is faithfulness.
- Recommended chunk coverage per city: freeway corridors, recurrent bottlenecks,
  signal-timing control capability + constraints, transit alternatives,
  incident-history summary, road capacities.

## What the advisor does with these
`knowledge_base.py` loads the file, builds a query from the explanation's node
names + regions, scores every chunk's `text`+`region_tags` against it, and
injects the top-k (`retrieval.top_k` in `configs/advisor.yaml`) as the
"CITY CONTEXT" block of the prompt.
