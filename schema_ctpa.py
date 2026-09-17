"""CTPA schema: every finding is a JSON boolean or null, plus a verbatim quote.

present: true  = explicitly described in the report body
         false = explicitly negated
         null  = not mentioned
The pe_* fields describe the embolism and must be null unless pulmonary_embolism is true.
To add a finding, add one line to ReportExtraction (name: Finding).
"""

import json

from pydantic import BaseModel, Field, field_validator, model_validator

from schema import blank_to_none


class Finding(BaseModel):
    present: bool | None
    evidence: str | None = Field(max_length=200)

    _blank_evidence = field_validator("evidence", mode="before")(blank_to_none)

    @model_validator(mode="after")
    def evidence_matches_present(self):
        if self.present is None and self.evidence is not None:
            raise ValueError("evidence must be null when present is null (not mentioned)")
        if self.present is not None and self.evidence is None:
            raise ValueError("evidence (a verbatim quote from the report) is required when present is true or false")
        return self


PE_SUBFIELDS = ["pe_acute", "pe_chronic", "pe_saddle", "pe_main", "pe_lobar", "pe_segmental", "pe_subsegmental",
                "pe_right", "pe_left", "pe_multiple", "pe_occlusive", "pe_nonocclusive"]


class ReportExtraction(BaseModel):
    suboptimal_study: Finding
    poor_contrast_opacification: Finding
    motion_artifact: Finding
    pulmonary_embolism: Finding
    pe_acute: Finding
    pe_chronic: Finding
    pe_saddle: Finding
    pe_main: Finding
    pe_lobar: Finding
    pe_segmental: Finding
    pe_subsegmental: Finding
    pe_right: Finding
    pe_left: Finding
    pe_multiple: Finding
    pe_occlusive: Finding
    pe_nonocclusive: Finding
    right_heart_strain: Finding
    pulmonary_artery_enlargement: Finding
    perfusion_defect: Finding
    pulmonary_infarct: Finding
    consolidation: Finding
    ground_glass_opacity: Finding
    atelectasis: Finding
    pulmonary_nodule: Finding
    emphysema: Finding
    mosaic_attenuation: Finding
    pleural_effusion: Finding
    pneumothorax: Finding
    pericardial_effusion: Finding
    cardiomegaly: Finding
    lymphadenopathy: Finding

    @model_validator(mode="after")
    def embolism_subfields_need_embolism(self):
        if self.pulmonary_embolism.present is not True:
            filled = [name for name in PE_SUBFIELDS if getattr(self, name).present is not None]
            if filled:
                raise ValueError(f"{', '.join(filled)}: the pe_ fields must be null (present null, evidence null) unless pulmonary_embolism is true")
        return self


FINDING_NAMES = [name for name, field in ReportExtraction.model_fields.items() if field.annotation is Finding]
FLAT_COLUMNS = [f"{name}_{suffix}" for name in FINDING_NAMES for suffix in ("present", "evidence")]
SCORED_FIELDS = [f"{name}_present" for name in FINDING_NAMES]
NUMERIC_FIELDS = set()
REPORTS_FILE = "reports_ctpa.csv"
GROUND_TRUTH_FILE = "ground_truth_ctpa.csv"

EXTRACTION_PROMPT = """You extract findings from one CT pulmonary angiogram (CTPA) report. Reply with JSON only.

Every field has "present":
- true: the finding is explicitly described in the report body (FINDINGS, IMPRESSION or the narrative).
- false: the report explicitly negates it ("no", "without", "not identified", "negative for", "resolved", "normal", "unremarkable").
- null: the report says nothing about it.

Rules:
- Text in INDICATION, HISTORY or COMPARISON never counts. The three study-quality fields may be taken from TECHNIQUE; nothing else may.
- Never infer. A described finding with a hedged interpretation ("likely", "may represent", "favored", "suspicious for") is true. "Cannot be excluded" or "not excluded" alone is never true. A measurement alone is not a finding.
- "Stable", "unchanged", "decreased", "residual" findings are true; "resolved" is false.
- A whole structure described as normal or clear negates its findings: "the lungs are clear" makes consolidation, ground_glass_opacity, atelectasis, pulmonary_nodule, emphysema, mosaic_attenuation and pulmonary_infarct false.
- pulmonary_embolism is true for any acute or chronic embolus, thrombus or filling defect in a pulmonary artery. "No acute pulmonary embolism" together with chronic thrombus means pulmonary_embolism true, pe_acute false, pe_chronic true.
- pe_acute is true only when the report itself calls the embolism acute, false when it says there is no acute embolism, otherwise null. pe_chronic likewise needs the word chronic (or chronic-appearing). Words such as new, residual, resolving or single do not decide acute or chronic.
- All pe_ fields must be null (present null, evidence null) unless pulmonary_embolism is true. When it is true: pe_saddle, pe_main, pe_lobar (interlobar counts as lobar), pe_segmental and pe_subsegmental are the arterial levels, true for each level described and false for a level explicitly negated; a segmental artery of a lobe ("right lower lobe segmental artery") is segmental, not lobar, and pe_lobar needs the lobar or interlobar artery itself; pe_right and pe_left are the sides involved ("bilateral" makes both true), false only when the report explicitly says that side is clear or free of thrombus; pe_multiple is true when more than one embolus, filling defect or vessel is involved and false for a single one; a clot described as occlusive makes pe_occlusive true and pe_nonocclusive false, a clot described as nonocclusive the reverse, and both are true when both are described.
- suboptimal_study is true when the report calls the study limited, suboptimal, degraded or nondiagnostic for pulmonary embolism, false when the opacification is called adequate, good or excellent and no artifact is said to limit evaluation; an artifact described as mild or as not limiting evaluation does not make the study suboptimal. motion_artifact is true when any motion artifact is mentioned and false when the report says there is none. poor_contrast_opacification is true when opacification is called poor, suboptimal or inadequate and false when it is called adequate, good or excellent.
- right_heart_strain is true when the report describes right ventricular dilation or enlargement, an RV/LV ratio above 1, septal flattening or bowing, contrast reflux into the IVC, or calls it right heart strain; false when it states no strain or a normal right ventricle. pulmonary_artery_enlargement: the pulmonary artery described as enlarged or dilated. perfusion_defect: a perfusion or iodine-map defect. pulmonary_infarct: an infarct described. lymphadenopathy covers mediastinal, hilar or axillary nodes.

"evidence": one contiguous span copied exactly from the report (max 200 characters, no "..." and no paraphrase) supporting the value. Required when present is true or false; null when present is null."""


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
        finding = getattr(extraction, name)
        row[f"{name}_present"] = finding.present
        row[f"{name}_evidence"] = finding.evidence
    return row


def normalize(data: dict) -> tuple[dict, dict]:
    """Repair what constrained decoding cannot enforce, so that only a true finding without a quote triggers a retry.

    false without a quote becomes null (a negation must be quoted), a quote on a null finding is dropped,
    pe_ fields are cleared when there is no embolism, and quotes longer than 200 characters are cut (a prefix of a
    verbatim quote is still verbatim). Returns the repaired data and the counts of repairs made.
    """
    counts = {"false_without_quote_to_null": 0, "quote_dropped_for_null": 0, "pe_fields_cleared": 0, "quote_truncated": 0}
    if not isinstance(data, dict):
        return data, {}
    for name in FINDING_NAMES:
        finding = data.get(name)
        if not isinstance(finding, dict):
            continue
        evidence = finding.get("evidence")
        if isinstance(evidence, str) and evidence.strip() == "":
            evidence = None
            finding["evidence"] = None
        if isinstance(evidence, str) and len(evidence) > 200:
            finding["evidence"] = evidence[:200]
            counts["quote_truncated"] += 1
        if finding.get("present") is False and evidence is None:
            finding["present"] = None
            counts["false_without_quote_to_null"] += 1
        elif finding.get("present") is None and evidence is not None:
            finding["evidence"] = None
            counts["quote_dropped_for_null"] += 1
    embolism = data.get("pulmonary_embolism")
    if not (isinstance(embolism, dict) and embolism.get("present") is True):
        for name in PE_SUBFIELDS:
            finding = data.get(name)
            if isinstance(finding, dict) and (finding.get("present") is not None or finding.get("evidence") is not None):
                finding["present"] = None
                finding["evidence"] = None
                counts["pe_fields_cleared"] += 1
    return data, {key: value for key, value in counts.items() if value}
