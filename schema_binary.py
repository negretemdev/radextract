"""Binary schema: one true/false per finding, plus a verbatim quote that is not part of any agreement check.

true = the abnormality is explicitly described as present in the report body.
false = the report negates it or does not mention it.
To add a finding, add one line to ReportExtraction (name: BinaryObservation).
"""

import json

from pydantic import BaseModel, Field, field_validator, model_validator

from schema import blank_to_none


class BinaryObservation(BaseModel):
    present: bool
    evidence: str | None = Field(max_length=200)

    _blank_evidence = field_validator("evidence", mode="before")(blank_to_none)

    @model_validator(mode="after")
    def evidence_required_when_present(self):
        if self.present and self.evidence is None:
            raise ValueError("evidence (a verbatim quote from the report) is required when present is true")
        return self


class ReportExtraction(BaseModel):
    pulmonary_nodule: BinaryObservation
    ground_glass_opacity: BinaryObservation
    consolidation: BinaryObservation
    atelectasis: BinaryObservation
    emphysema: BinaryObservation
    bronchiectasis: BinaryObservation
    pleural_effusion: BinaryObservation
    pneumothorax: BinaryObservation
    pericardial_effusion: BinaryObservation
    cardiomegaly: BinaryObservation
    coronary_calcification: BinaryObservation
    mediastinal_lymphadenopathy: BinaryObservation
    hilar_lymphadenopathy: BinaryObservation
    aortic_dilation_or_aneurysm: BinaryObservation
    pulmonary_embolism: BinaryObservation
    hepatic_lesion: BinaryObservation
    adrenal_nodule: BinaryObservation
    thyroid_nodule: BinaryObservation
    osseous_lesion: BinaryObservation
    rib_fracture: BinaryObservation
    follow_up_recommended: BinaryObservation


FINDING_NAMES = [name for name, field in ReportExtraction.model_fields.items() if field.annotation is BinaryObservation]
FLAT_COLUMNS = [f"{name}_{suffix}" for name in FINDING_NAMES for suffix in ("present", "evidence")]
SCORED_FIELDS = [f"{name}_present" for name in FINDING_NAMES]
NUMERIC_FIELDS = set()
GROUND_TRUTH_FILE = "ground_truth_binary.csv"

EXTRACTION_PROMPT = """You extract findings from one chest CT radiology report. Reply with JSON only.

For each field, "present" is true only if the abnormality is explicitly described as present in the report body (FINDINGS, IMPRESSION or the narrative). It is false if the report negates it or does not mention it.
- Text in the EXAM, INDICATION, HISTORY, TECHNIQUE or COMPARISON lines never counts.
- Never infer. A described abnormality with a hedged interpretation ("likely", "may represent", "suspicious for") is true. "Cannot be excluded" alone is false.
- "Stable", "unchanged", "improved" or "residual" abnormalities are true. "Resolved" is false.
- pulmonary_nodule includes calcified granulomas and pulmonary masses. ground_glass_opacity is non-nodular ground-glass opacity; a ground-glass nodule counts only as pulmonary_nodule.
- hepatic_lesion, adrenal_nodule and thyroid_nodule are any focal lesion of that organ, including cysts. osseous_lesion is a focal bone lesion; degenerative change is not one.
- follow_up_recommended is true when the report recommends follow-up imaging or a procedure.

"evidence": when present is true, copy the phrase describing the finding exactly as written, as one contiguous span (verbatim, max 200 characters, no "..." and no paraphrase). When present is false, copy the negating phrase if the report has one, otherwise null."""


def build_messages(report_text: str, include_schema: bool) -> list[dict]:
    system_prompt = EXTRACTION_PROMPT
    if include_schema:
        system_prompt += "\n\nThe JSON object must match this JSON schema exactly:\n" + json.dumps(ReportExtraction.model_json_schema())
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Extract the findings from this report:\n\n" + report_text},
    ]


def flatten(extraction: ReportExtraction) -> dict:
    row = {}
    for name in FINDING_NAMES:
        observation = getattr(extraction, name)
        row[f"{name}_present"] = observation.present
        row[f"{name}_evidence"] = observation.evidence
    return row
