"""CTPA schema: every finding is two booleans plus a verbatim quote.

mentioned: true when the report addresses the finding at all (describes it or negates it), false when silent.
present:   true when the finding is described as present, false otherwise (negated or not mentioned).
The pe_* fields describe the embolism and are all false when pulmonary_embolism is not present.
To add a finding, add one line to ReportExtraction (name: Finding).
"""

import json

from pydantic import BaseModel, Field, field_validator, model_validator

from schema import blank_to_none


class Finding(BaseModel):
    mentioned: bool
    present: bool
    evidence: str | None = Field(max_length=200)

    _blank_evidence = field_validator("evidence", mode="before")(blank_to_none)

    @model_validator(mode="after")
    def present_needs_quote(self):
        if self.present and self.evidence is None:
            raise ValueError("evidence (a verbatim quote from the report) is required when present is true")
        return self


PE_SUBFIELDS = ["pe_acute", "pe_chronic", "pe_saddle", "pe_main", "pe_lobar", "pe_segmental", "pe_subsegmental",
                "pe_right", "pe_left", "pe_multiple", "pe_occlusive", "pe_nonocclusive"]
# one boolean per artery; a present artery implies its lobe's segmental field, its level field and its side
ARTERY_IMPLIES = {
    "pe_main_right": ["pe_main", "pe_right"], "pe_main_left": ["pe_main", "pe_left"],
    "pe_lobar_right_upper": ["pe_lobar", "pe_right"], "pe_lobar_right_middle": ["pe_lobar", "pe_right"],
    "pe_lobar_right_lower": ["pe_lobar", "pe_right"], "pe_lobar_left_upper": ["pe_lobar", "pe_left"],
    "pe_lobar_left_lower": ["pe_lobar", "pe_left"], "pe_lobar_interlobar_right": ["pe_lobar", "pe_right"],
    "pe_lobar_interlobar_left": ["pe_lobar", "pe_left"],
    "pe_segmental_right_upper": ["pe_segmental", "pe_right"], "pe_segmental_right_middle": ["pe_segmental", "pe_right"],
    "pe_segmental_right_lower": ["pe_segmental", "pe_right"], "pe_segmental_left_upper": ["pe_segmental", "pe_left"],
    "pe_segmental_lingula": ["pe_segmental", "pe_left"], "pe_segmental_left_lower": ["pe_segmental", "pe_left"],
    "pe_segment_rul_apical": ["pe_segmental_right_upper"], "pe_segment_rul_anterior": ["pe_segmental_right_upper"],
    "pe_segment_rul_posterior": ["pe_segmental_right_upper"], "pe_segment_rml_medial": ["pe_segmental_right_middle"],
    "pe_segment_rml_lateral": ["pe_segmental_right_middle"], "pe_segment_rll_superior": ["pe_segmental_right_lower"],
    "pe_segment_rll_medial_basal": ["pe_segmental_right_lower"], "pe_segment_rll_anterior_basal": ["pe_segmental_right_lower"],
    "pe_segment_rll_lateral_basal": ["pe_segmental_right_lower"], "pe_segment_rll_posterior_basal": ["pe_segmental_right_lower"],
    "pe_segment_lul_apicoposterior": ["pe_segmental_left_upper"], "pe_segment_lul_anterior": ["pe_segmental_left_upper"],
    "pe_segment_lingula_superior": ["pe_segmental_lingula"], "pe_segment_lingula_inferior": ["pe_segmental_lingula"],
    "pe_segment_lll_superior": ["pe_segmental_left_lower"], "pe_segment_lll_anteromedial_basal": ["pe_segmental_left_lower"],
    "pe_segment_lll_lateral_basal": ["pe_segmental_left_lower"], "pe_segment_lll_posterior_basal": ["pe_segmental_left_lower"],
}
PE_SUBFIELDS += list(ARTERY_IMPLIES)


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
    pe_main_right: Finding
    pe_main_left: Finding
    pe_lobar_right_upper: Finding
    pe_lobar_right_middle: Finding
    pe_lobar_right_lower: Finding
    pe_lobar_left_upper: Finding
    pe_lobar_left_lower: Finding
    pe_lobar_interlobar_right: Finding
    pe_lobar_interlobar_left: Finding
    pe_segmental_right_upper: Finding
    pe_segmental_right_middle: Finding
    pe_segmental_right_lower: Finding
    pe_segmental_left_upper: Finding
    pe_segmental_lingula: Finding
    pe_segmental_left_lower: Finding
    pe_segment_rul_apical: Finding
    pe_segment_rul_anterior: Finding
    pe_segment_rul_posterior: Finding
    pe_segment_rml_medial: Finding
    pe_segment_rml_lateral: Finding
    pe_segment_rll_superior: Finding
    pe_segment_rll_medial_basal: Finding
    pe_segment_rll_anterior_basal: Finding
    pe_segment_rll_lateral_basal: Finding
    pe_segment_rll_posterior_basal: Finding
    pe_segment_lul_apicoposterior: Finding
    pe_segment_lul_anterior: Finding
    pe_segment_lingula_superior: Finding
    pe_segment_lingula_inferior: Finding
    pe_segment_lll_superior: Finding
    pe_segment_lll_anteromedial_basal: Finding
    pe_segment_lll_lateral_basal: Finding
    pe_segment_lll_posterior_basal: Finding
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


FINDING_NAMES = [name for name, field in ReportExtraction.model_fields.items() if field.annotation is Finding]
FLAT_COLUMNS = [f"{name}_{suffix}" for name in FINDING_NAMES for suffix in ("mentioned", "present", "evidence")]
SCORED_FIELDS = [f"{name}_present" for name in FINDING_NAMES]      # agreement and review are judged on these
SECONDARY_FIELDS = [f"{name}_mentioned" for name in FINDING_NAMES]  # scored separately, never a review trigger
NUMERIC_FIELDS = set()
REPORTS_FILE = "reports_ctpa.csv"
GROUND_TRUTH_FILE = "ground_truth_ctpa.csv"
RAW_DIR_NAME = "ctpa_v2"

EXTRACTION_PROMPT = """You extract findings from one CT pulmonary angiogram (CTPA) report. Reply with JSON only.

Every field has two booleans:
- "mentioned": true when the report addresses the finding at all, either describing it or explicitly negating it ("no", "without", "not identified", "negative for", "resolved", "normal", "unremarkable"); false when the report says nothing about it.
- "present": true only when the finding is explicitly described as present in the report body (FINDINGS, IMPRESSION or the narrative); false when it is negated or not mentioned.
Example for pleural_effusion: "Small left pleural effusion" gives mentioned true, present true; "No pleural effusion" gives mentioned true, present false; a report that never speaks of effusions gives mentioned false, present false. So present true always comes with mentioned true, and mentioned true with present false is exactly a negation.

Rules:
- Text in INDICATION, HISTORY or COMPARISON never counts. The three study-quality fields may be taken from TECHNIQUE; nothing else may.
- Never infer. A described finding with a hedged interpretation ("likely", "may represent", "favored", "suspicious for") is true. "Cannot be excluded" or "not excluded" alone is never true. A measurement alone is not a finding.
- "Stable", "unchanged", "decreased", "residual" findings are true; "resolved" is false.
- A whole structure described as normal or clear negates its findings: "the lungs are clear" makes consolidation, ground_glass_opacity, atelectasis, pulmonary_nodule, emphysema, mosaic_attenuation and pulmonary_infarct mentioned but not present.
- pulmonary_embolism is true for any acute or chronic embolus, thrombus or filling defect in a pulmonary artery. "No acute pulmonary embolism" together with chronic thrombus means pulmonary_embolism present, pe_acute mentioned but not present, pe_chronic present.
- pe_acute is present only when the report itself calls the embolism acute; mentioned when it says acute or no acute. pe_chronic likewise needs the word chronic (or chronic-appearing). Words such as new, residual, resolving or single do not decide acute or chronic.
- All pe_ fields are mentioned false, present false, evidence null unless pulmonary_embolism is present. When it is: pe_saddle, pe_main, pe_lobar, pe_segmental and pe_subsegmental are the arterial levels, present for each level described, mentioned but not present for a level explicitly negated. The lobar arteries are the upper, middle and lower lobe pulmonary arteries and the interlobar arteries: "right lower lobe pulmonary artery", "left upper lobe artery", "lower lobe arteries" and "right interlobar artery" all make pe_lobar present. A segmental artery of a lobe ("right lower lobe segmental artery", "segmental branches") is segmental, not lobar; pe_right and pe_left are the sides involved ("bilateral" makes both true), mentioned but not present only when the report explicitly says that side is clear or free of thrombus; pe_multiple is true when more than one embolus, filling defect or vessel is involved and false for a single one; a clot described as occlusive makes pe_occlusive true and pe_nonocclusive false, a clot described as nonocclusive the reverse, and both are true when both are described.
- Artery fields, all present only when the report names that artery as involved: pe_main_right and pe_main_left (the right and left main pulmonary arteries); pe_lobar_<lobe> for the right upper, right middle, right lower, left upper and left lower lobe pulmonary arteries ("right lower lobe pulmonary artery", "left upper lobe artery", "lower lobe arteries") and pe_lobar_interlobar_right/left; pe_segmental_<lobe> when segmental arteries of that lobe are involved ("right lower lobe segmental arteries", "segmental branches of both lower lobes"), with the lingula separate from the left upper lobe; pe_segment_<lobe>_<segment> only when the report names the segment ("posterior basal segmental artery of the right lower lobe"). Unnamed lobes stay unmentioned ("bilateral lobar clot", "bilateral segmental branches" and "segmental arteries of both lungs" do not name a lobe), while "all lobes" names every lobe including the lingula. A whole side described as clear or patent makes every artery field of that side mentioned but not present.
- suboptimal_study is present when the report calls the study limited, suboptimal, degraded or nondiagnostic for pulmonary embolism; mentioned but not present when the report speaks of the study's diagnostic quality or says an artifact does not limit evaluation; not mentioned when the report only comments on opacification or artifact without judging the study. An artifact described as mild does not make the study suboptimal. motion_artifact is present when any motion artifact is described. poor_contrast_opacification is present when opacification is called poor, suboptimal or inadequate, and mentioned but not present when it is called adequate, good or excellent.
- right_heart_strain is present when the report describes right ventricular dilation or enlargement, an RV/LV ratio above 1, septal flattening or bowing, contrast reflux into the IVC, or calls it right heart strain; mentioned but not present when it states no strain or a normal right ventricle. pulmonary_artery_enlargement: the pulmonary artery described as enlarged or dilated. perfusion_defect: a perfusion or iodine-map defect. pulmonary_infarct: an infarct described. lymphadenopathy covers mediastinal, hilar or axillary nodes.

"evidence": one contiguous span copied exactly from the report (max 200 characters, no "..." and no paraphrase). Required when present is true (the phrase describing the finding); when mentioned but not present, the negating phrase; null when not mentioned."""


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
        row[f"{name}_mentioned"] = finding.mentioned
        row[f"{name}_present"] = finding.present
        row[f"{name}_evidence"] = finding.evidence
    return row


def normalize(data: dict) -> tuple[dict, dict]:
    """Repair what constrained decoding cannot enforce, so that only a present finding without a quote triggers a retry.

    present implies mentioned; a negation without a quote is treated as not mentioned; a quote on an unmentioned
    finding is dropped; pe_ fields are cleared when there is no embolism; quotes over 200 characters are cut
    (a prefix of a verbatim quote is still verbatim). Returns the repaired data and the counts of repairs made.
    """
    counts = {"present_made_mentioned": 0, "negation_without_quote_unmentioned": 0, "quote_dropped_for_unmentioned": 0,
              "pe_fields_cleared": 0, "quote_truncated": 0}
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
        if finding.get("present") is True and finding.get("mentioned") is not True:
            finding["mentioned"] = True
            counts["present_made_mentioned"] += 1
        if finding.get("mentioned") is True and finding.get("present") is not True and evidence is None:
            finding["mentioned"] = False
            counts["negation_without_quote_unmentioned"] += 1
        if finding.get("mentioned") is False and finding.get("present") is not True and evidence is not None:
            finding["evidence"] = None
            counts["quote_dropped_for_unmentioned"] += 1
    counts["level_made_present"] = 0
    for artery, implied_names in ARTERY_IMPLIES.items():
        finding = data.get(artery)
        if isinstance(finding, dict) and finding.get("present") is True:
            queue = list(implied_names)
            while queue:
                name = queue.pop()
                implied = data.get(name)
                if isinstance(implied, dict) and implied.get("present") is not True:
                    implied["present"] = True
                    implied["mentioned"] = True
                    if implied.get("evidence") is None:
                        implied["evidence"] = finding.get("evidence")
                    counts["level_made_present"] += 1
                queue += ARTERY_IMPLIES.get(name, [])
    embolism = data.get("pulmonary_embolism")
    if not (isinstance(embolism, dict) and embolism.get("present") is True):
        for name in PE_SUBFIELDS:
            finding = data.get(name)
            if isinstance(finding, dict) and (finding.get("mentioned") or finding.get("present") or finding.get("evidence") is not None):
                finding["mentioned"] = False
                finding["present"] = False
                finding["evidence"] = None
                counts["pe_fields_cleared"] += 1
    return data, {key: value for key, value in counts.items() if value}
