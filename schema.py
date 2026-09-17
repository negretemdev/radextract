"""Pydantic schema and extraction prompt for chest CT report extraction.

Every asserted value must be traceable to a verbatim quote from the report, so
each Observation carries an evidence string and the validators enforce that the
quote is there whenever a status is asserted.

To add a finding, add one line to ReportExtraction (name: Observation).
extract.py and evaluate.py discover the Observation fields at runtime.
"""

import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Status = Literal["present", "absent", "not_mentioned"]
Laterality = Literal["left", "right", "not_stated"]
Lobe = Literal["right_upper", "right_middle", "right_lower", "left_upper", "left_lower", "lingula", "not_stated"]
Attenuation = Literal["solid", "part_solid", "ground_glass", "not_stated"]
YesNo = Literal["yes", "no", "not_stated"]
Margins = Literal["smooth", "lobulated", "spiculated", "irregular", "not_stated"]
Modality = Literal["ct", "pet_ct", "mri", "biopsy", "other", "not_stated"]


def blank_to_none(value):
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


class Observation(BaseModel):
    status: Status
    evidence: str | None = Field(max_length=200)

    _blank_evidence = field_validator("evidence", mode="before")(blank_to_none)

    @model_validator(mode="after")
    def evidence_matches_status(self):
        if self.status == "not_mentioned" and self.evidence is not None:
            raise ValueError("evidence must be null when status is not_mentioned")
        if self.status != "not_mentioned" and self.evidence is None:
            raise ValueError(f"evidence (a verbatim quote from the report) is required when status is {self.status}")
        return self


class LargestNodule(BaseModel):
    size_mm: float | None
    laterality: Laterality
    lobe: Lobe
    attenuation: Attenuation
    calcified: YesNo
    margins: Margins
    evidence: str | None = Field(max_length=200)

    _blank_evidence = field_validator("evidence", mode="before")(blank_to_none)


class FollowUp(BaseModel):
    recommended: YesNo
    modality: Modality
    interval_months: float | None
    evidence: str | None = Field(max_length=200)

    _blank_evidence = field_validator("evidence", mode="before")(blank_to_none)

    @model_validator(mode="after")
    def evidence_matches_recommendation(self):
        if self.recommended == "not_stated":
            if self.evidence is not None or self.modality != "not_stated" or self.interval_months is not None:
                raise ValueError("when follow_up.recommended is not_stated, modality must be not_stated and interval_months and evidence must be null")
        elif self.evidence is None:
            raise ValueError("follow_up.evidence (a verbatim quote) is required when recommended is yes or no")
        return self


class ReportExtraction(BaseModel):
    pulmonary_nodule: Observation
    ground_glass_opacity: Observation
    consolidation: Observation
    atelectasis: Observation
    emphysema: Observation
    bronchiectasis: Observation
    pleural_effusion: Observation
    pneumothorax: Observation
    pericardial_effusion: Observation
    cardiomegaly: Observation
    coronary_calcification: Observation
    mediastinal_lymphadenopathy: Observation
    hilar_lymphadenopathy: Observation
    aortic_dilation_or_aneurysm: Observation
    pulmonary_embolism: Observation
    hepatic_lesion: Observation
    adrenal_nodule: Observation
    thyroid_nodule: Observation
    osseous_lesion: Observation
    rib_fracture: Observation
    nodule_count: int = Field(ge=0)
    largest_nodule: LargestNodule
    follow_up: FollowUp

    @model_validator(mode="after")
    def nodule_fields_are_consistent(self):
        if self.pulmonary_nodule.status == "present" and self.nodule_count == 0:
            raise ValueError("nodule_count must be greater than 0 when pulmonary_nodule is present")
        if self.pulmonary_nodule.status != "present" and self.nodule_count != 0:
            raise ValueError("nodule_count must be 0 unless pulmonary_nodule is present")
        largest = self.largest_nodule
        if self.nodule_count == 0:
            literal_values = {largest.laterality, largest.lobe, largest.attenuation, largest.calcified, largest.margins}
            if largest.size_mm is not None or largest.evidence is not None or literal_values != {"not_stated"}:
                raise ValueError("largest_nodule must be empty (size_mm and evidence null, everything else not_stated) when nodule_count is 0")
        elif largest.evidence is None:
            raise ValueError("largest_nodule.evidence (a verbatim quote) is required when nodules are present")
        return self


FINDING_NAMES = [name for name, field in ReportExtraction.model_fields.items() if field.annotation is Observation]

EXTRACTION_PROMPT = """You extract structured findings from one chest CT radiology report. Reply with JSON only.

Status of each finding:
- "present": the abnormality is explicitly described in the report body (FINDINGS, IMPRESSION or the narrative).
- "absent": the report explicitly negates it ("no", "without", "not identified", "resolved", "unremarkable", "within normal limits", "no significant"). A whole structure described as normal, clear or unremarkable negates its findings ("the lungs are clear" negates all lung findings; "the thyroid gland is unremarkable" negates thyroid_nodule).
- "not_mentioned": the report says nothing about it.
- Text in the EXAM, INDICATION, HISTORY, TECHNIQUE or COMPARISON lines never counts as a finding.
- Never infer. A described abnormality with a hedged interpretation ("likely", "may represent", "suspicious for") is present. "Cannot be excluded" alone never makes a finding present.
- "Stable" or "unchanged" abnormalities are present.
- osseous_lesion means a focal bone lesion; degenerative change is not a lesion. hepatic_lesion, adrenal_nodule and thyroid_nodule cover any focal lesion of that organ, including cysts.

Evidence: copy the supporting phrase exactly as written in the report, as one contiguous span (verbatim, max 200 characters, no "..." and no paraphrase). Use null only when status is "not_mentioned".

Nodules: a calcified granuloma (calcified "yes") or a pulmonary mass counts as a pulmonary nodule. ground_glass_opacity is for non-nodular ground-glass opacities; a ground-glass nodule goes under pulmonary_nodule with attenuation "ground_glass". nodule_count is the number of pulmonary nodules described (0 if none). largest_nodule describes the largest pulmonary nodule; use null or "not_stated" for details the report does not give, and leave it empty when there are no nodules.

Sizes are in mm (1 cm = 10 mm). For "a x b" measurements use the largest dimension.

follow_up: any recommended follow-up imaging or procedure ("no" when the report states that no follow-up is needed). interval_months is the recommended interval in months (null if none is given)."""


def build_messages(report_text: str, include_schema: bool) -> list[dict]:
    """Chat messages for one report. include_schema adds the JSON schema for models run without constrained format."""
    system_prompt = EXTRACTION_PROMPT
    if include_schema:
        system_prompt += "\n\nThe JSON object must match this JSON schema exactly:\n" + json.dumps(ReportExtraction.model_json_schema())
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Extract the findings from this report:\n\n" + report_text},
    ]


FLAT_COLUMNS = [f"{name}_{suffix}" for name in FINDING_NAMES for suffix in ("status", "evidence")]
FLAT_COLUMNS.append("nodule_count")
FLAT_COLUMNS += [f"largest_nodule_{field}" for field in LargestNodule.model_fields]
FLAT_COLUMNS += [f"follow_up_{field}" for field in FollowUp.model_fields]
SCORED_FIELDS = [column for column in FLAT_COLUMNS if not column.endswith("_evidence")]
NUMERIC_FIELDS = {"nodule_count", "largest_nodule_size_mm", "follow_up_interval_months"}
REPORTS_FILE = "reports.csv"
GROUND_TRUTH_FILE = "ground_truth.csv"


def flatten(extraction: ReportExtraction) -> dict:
    """One flat row per extraction, in FLAT_COLUMNS order."""
    row = {}
    for name in FINDING_NAMES:
        observation = getattr(extraction, name)
        row[f"{name}_status"] = observation.status
        row[f"{name}_evidence"] = observation.evidence
    row["nodule_count"] = extraction.nodule_count
    for prefix, part in (("largest_nodule", extraction.largest_nodule), ("follow_up", extraction.follow_up)):
        for field, value in part.model_dump().items():
            row[f"{prefix}_{field}"] = value
    return row
