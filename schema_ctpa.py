"""CTPA schema: every finding is two booleans plus a verbatim quote.

mentioned: true when the report addresses the finding at all (describes it or negates it), false when silent.
present:   true when the finding is described as present, false otherwise (negated or not mentioned).
The pe_* fields describe the embolism and are all false when pulmonary_embolism is not present.
To add a finding, add one line to ReportExtraction (name: Finding).
"""

import json
import re

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
KEY_FINDING = "pulmonary_embolism_present"   # evaluate.py reports report-level correctness on the key fields, and among reports where this is present
KEY_FIELDS = [f"{name}_present" for name in FINDING_NAMES if name == "pulmonary_embolism" or name.startswith("pe_")]

LOBE_WORDS = {"right_upper": "right upper lobe", "right_middle": "right middle lobe", "right_lower": "right lower lobe",
              "left_upper": "left upper lobe", "left_lower": "left lower lobe", "lingula": "lingula"}
SEGMENT_WORDS = {"rul": "right upper lobe", "rml": "right middle lobe", "rll": "right lower lobe", "lul": "left upper lobe",
                 "lingula": "lingula", "lll": "left lower lobe"}
FIELD_DEFINITIONS = {
    "suboptimal_study": "the study is called limited, suboptimal, degraded or nondiagnostic for pulmonary embolism",
    "poor_contrast_opacification": "opacification of the pulmonary arteries is called poor, suboptimal or inadequate",
    "motion_artifact": "respiratory or motion artifact is described",
    "pulmonary_embolism": "any acute or chronic embolus, thrombus or filling defect in a pulmonary artery",
    "pe_acute": "the embolism is called acute by the report",
    "pe_chronic": "the embolism is called chronic or chronic-appearing (webs, mural or calcified thrombus)",
    "pe_saddle": "a saddle embolus at the bifurcation of the main pulmonary artery",
    "pe_main": "embolus in the right or left main pulmonary artery",
    "pe_lobar": "embolus in a lobar artery: an upper, middle or lower lobe pulmonary artery, or an interlobar artery",
    "pe_segmental": "embolus in a segmental artery or segmental branch (a subsegmental artery does not count)",
    "pe_subsegmental": "embolus in a subsegmental artery or branch",
    "pe_right": "embolus in a right-sided pulmonary artery",
    "pe_left": "embolus in a left-sided pulmonary artery",
    "pe_multiple": "clot in more than one anatomically distinct artery, or more than one embolus, thrombus or filling defect named (plural, multiple, several, a count above one); a thrombus extending from a parent artery into its branches counts as multiple; false for clot confined to one vessel or called single",
    "pe_occlusive": "clot described as occlusive",
    "pe_nonocclusive": "clot described as nonocclusive",
    "right_heart_strain": "right ventricular dilation or enlargement, RV/LV ratio above 1, septal flattening or bowing, contrast reflux into the IVC, or the words right heart strain",
    "pulmonary_artery_enlargement": "the pulmonary artery described as enlarged or dilated (a measurement alone does not count)",
    "perfusion_defect": "a perfusion or iodine-map defect",
    "pulmonary_infarct": "a pulmonary infarct described, including a hedged one (infarct versus pneumonia)",
    "consolidation": "consolidation in the lung",
    "ground_glass_opacity": "ground-glass opacity in the lung",
    "atelectasis": "atelectasis",
    "pulmonary_nodule": "a pulmonary nodule, including cavitary or septic nodules",
    "emphysema": "emphysema",
    "mosaic_attenuation": "mosaic attenuation of the lung parenchyma",
    "pleural_effusion": "pleural effusion or pleural fluid",
    "pneumothorax": "pneumothorax",
    "pericardial_effusion": "pericardial effusion or pericardial fluid",
    "cardiomegaly": "the heart or a chamber described as enlarged",
    "lymphadenopathy": "enlarged mediastinal, hilar or axillary lymph nodes",
    "pe_main_right": "embolus in the right main pulmonary artery",
    "pe_main_left": "embolus in the left main pulmonary artery",
    "pe_lobar_interlobar_right": "embolus in the right interlobar pulmonary artery",
    "pe_lobar_interlobar_left": "embolus in the left interlobar pulmonary artery",
}
for key, words in LOBE_WORDS.items():
    if key != "lingula":
        FIELD_DEFINITIONS[f"pe_lobar_{key}"] = f"embolus in the {words} pulmonary artery itself (the lobar artery, not its segmental branches)"
    FIELD_DEFINITIONS[f"pe_segmental_{key}"] = f"embolus in a segmental artery or segmental branch of the {words} (the lobe must be named by the report, directly or as 'both lower lobes', 'bilateral lower' or 'all lobes'; a subsegmental branch of the lobe does not count)"
FIELD_DEFINITIONS["pe_segmental_left_upper"] += "; the lingular segments belong to the left upper lobe, so a lingular artery counts here too"
for name in FINDING_NAMES:
    if name.startswith("pe_segment_"):
        lobe, segment = name[len("pe_segment_"):].split("_", 1)
        FIELD_DEFINITIONS[name] = f"embolus in the {segment.replace('_', ' ')} segmental artery of the {SEGMENT_WORDS[lobe]} (the segment must be named by the report)"
assert set(FIELD_DEFINITIONS) == set(FINDING_NAMES), set(FINDING_NAMES) ^ set(FIELD_DEFINITIONS)

# a lobe, side or segment field counts only when its own quote names it: each entry is a list of phrase groups,
# every group must be matched by at least one phrase (case-insensitive substring)
LOBE_PHRASES = {
    "right_upper": ["right upper", "rul", "r upper", "upper lobes", "both upper", "bilateral upper", "all lobes", "right and left upper", "right upper and"],
    "right_middle": ["right middle", "rml", "r middle", "middle lobe", "all lobes"],
    "right_lower": ["right lower", "rll", "r lower", "lower lobes", "both lower", "bilateral lower", "all lobes", "right and left lower", "right upper and lower", "right middle and lower"],
    "left_upper": ["left upper", "lul", "l upper", "upper lobes", "both upper", "bilateral upper", "all lobes", "lingula", "lingular", "right and left upper"],
    "left_lower": ["left lower", "lll", "l lower", "lower lobes", "both lower", "bilateral lower", "all lobes", "right and left lower", "left upper and lower"],
    "lingula": ["lingula", "lingular", "all lobes"],
}
# each segment needs every group matched: "anterior and lateral basal" names both the anterior basal and the lateral basal segments
SEGMENT_PHRASES = {"apical": [["apical"]], "anterior": [["anterior"]], "posterior": [["posterior"]], "medial": [["medial"]], "lateral": [["lateral"]],
                   "superior": [["superior"]], "inferior": [["inferior"]], "medial_basal": [["medial", "mediobasal"], ["basal"]],
                   "anterior_basal": [["anterior", "anterobasal"], ["basal"]], "lateral_basal": [["lateral", "laterobasal"], ["basal"]],
                   "posterior_basal": [["posterior", "posterobasal"], ["basal"]], "apicoposterior": [["apicoposterior", "apical-posterior"]],
                   "anteromedial_basal": [["anteromedial", "anterior medial"], ["basal"]]}
SEGMENT_LOBES = {"rul": "right_upper", "rml": "right_middle", "rll": "right_lower", "lul": "left_upper", "lingula": "lingula", "lll": "left_lower"}
QUOTE_MUST_CONTAIN = {
    "pe_main_right": [["right main", "right and left main", "both main", "bilateral main", "r main"]],
    "pe_main_left": [["left main", "right and left main", "both main", "bilateral main", "l main"]],
    "pe_lobar_interlobar_right": [["interlobar"]],
    "pe_lobar_interlobar_left": [["interlobar"]],
}
for lobe, phrases in LOBE_PHRASES.items():
    if lobe != "lingula":
        QUOTE_MUST_CONTAIN[f"pe_lobar_{lobe}"] = [phrases]
    QUOTE_MUST_CONTAIN[f"pe_segmental_{lobe}"] = [phrases]
for name in FINDING_NAMES:
    if name.startswith("pe_segment_"):
        lobe, segment = name[len("pe_segment_"):].split("_", 1)
        QUOTE_MUST_CONTAIN[name] = [LOBE_PHRASES[SEGMENT_LOBES[lobe]]] + SEGMENT_PHRASES[segment]


def question_messages(report_text: str, field: str, include_schema: bool, context: str = "") -> list[dict]:
    """Chat messages asking about one field only; the answer is a single Finding object.
    context: facts already extracted from the same report and the sibling fields answered separately (fine mode)."""
    system_prompt = (EXTRACTION_PROMPT + f"\n\nAnswer for exactly one field, {field}: {FIELD_DEFINITIONS[field]}. "
                     "Return one JSON object with the keys mentioned, present and evidence for this field only.")
    if context:
        system_prompt += "\n\n" + context
    if include_schema:
        system_prompt += "\n\nThe JSON object must match this JSON schema exactly:\n" + json.dumps(Finding.model_json_schema())
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Extract this one finding from this report:\n\n" + report_text},
    ]

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
- All pe_ fields are mentioned false, present false, evidence null unless pulmonary_embolism is present. When it is: pe_saddle, pe_main, pe_lobar, pe_segmental and pe_subsegmental are the arterial levels, present for each level described, mentioned but not present for a level explicitly negated. The lobar arteries are the upper, middle and lower lobe pulmonary arteries and the interlobar arteries: "right lower lobe pulmonary artery", "left upper lobe artery", "lower lobe arteries" and "right interlobar artery" all make pe_lobar present. A segmental artery of a lobe ("right lower lobe segmental artery", "segmental branches") is segmental, not lobar; pe_right and pe_left are the sides involved ("bilateral" makes both true), mentioned but not present only when the report explicitly says that side is clear or free of thrombus; pe_multiple is present when clot is described in more than one anatomically distinct artery, or when the report names more than one embolus, thrombus or filling defect (plural nouns, "multiple", "several", a count above one); a single thrombus extending from a parent artery into its branches still counts as multiple because more than one vessel is involved; mentioned but not present when the clot is confined to one vessel or called single ("a single filling defect"); unmentioned only when the report gives no indication of how many vessels are involved; a clot described as occlusive makes pe_occlusive true and pe_nonocclusive false, a clot described as nonocclusive the reverse, and both are true when both are described.
- Artery fields, all present only when the report names that artery as involved: pe_main_right and pe_main_left (the right and left main pulmonary arteries); pe_lobar_<lobe> for the right upper, right middle, right lower, left upper and left lower lobe pulmonary arteries ("right lower lobe pulmonary artery", "left upper lobe artery", "lower lobe arteries") and pe_lobar_interlobar_right/left; pe_segmental_<lobe> when segmental arteries of that lobe are involved ("right lower lobe segmental arteries", "segmental branches of both lower lobes"); the lingula is part of the left upper lobe, so a lingular artery sets both pe_segmental_lingula and pe_segmental_left_upper; pe_segment_<lobe>_<segment> only when the report names the segment ("posterior basal segmental artery of the right lower lobe"). Side words never name lobes: "bilateral", "both lungs", "right and left", "segmental branches of both lungs" set only pe_right, pe_left and the level field, and every pe_segmental_<lobe> stays unmentioned; only a named lobe, "both lower lobes" or "all lobes" (every lobe including the lingula) sets a per-lobe field. Collective phrases ("segmental branches", "multiple segmental and subsegmental branches", "all lobes") name lobes at most, never individual segments: all pe_segment_ fields stay unmentioned unless the segment is named. Levels never roll up or down: an embolus in a subsegmental branch sets pe_subsegmental only, never pe_segmental or any pe_segmental_<lobe>, even when the lobe is named ("a subsegmental branch of the right lower lobe"), and clot in a lobar artery does not set that lobe's segmental field. Never infer a vessel from the path a clot must have taken: in "thrombus in the right interlobar artery extending into the right middle lobe and right lower lobe segmental arteries", "right middle lobe" modifies "segmental arteries", so set pe_segmental_right_middle, not pe_lobar_right_middle. A saddle embolus sets pe_saddle; pe_main_right and pe_main_left only when the report says the clot extends into the main pulmonary arteries. A whole side described as clear or patent makes every artery field of that side mentioned but not present.
- suboptimal_study is present when the report calls the study limited, suboptimal, degraded or nondiagnostic for pulmonary embolism; mentioned but not present when the report speaks of the study's diagnostic quality or says an artifact does not limit evaluation; not mentioned when the report only comments on opacification or artifact without judging the study. An artifact described as mild does not make the study suboptimal. motion_artifact is present when any motion artifact is described. poor_contrast_opacification is present when opacification is called poor, suboptimal or inadequate, and mentioned but not present when it is called adequate, good or excellent.
- right_heart_strain is present when the report describes right ventricular dilation or enlargement, an RV/LV ratio above 1, septal flattening or bowing, contrast reflux into the IVC, or calls it right heart strain; mentioned but not present when it states no strain or a normal right ventricle. pulmonary_artery_enlargement: the pulmonary artery described as enlarged or dilated. perfusion_defect: a perfusion or iodine-map defect. pulmonary_infarct: an infarct described. lymphadenopathy covers mediastinal, hilar or axillary nodes.

"evidence": one contiguous span copied exactly from the report (max 200 characters, no "..." and no paraphrase). Required when present is true (the phrase describing the finding); when mentioned but not present, the negating phrase; null when not mentioned. For an artery field quote the phrase that names it; "both", "bilateral", "right and left" name both sides, and "all lobes" or "both lower lobes" name those lobes, so quote those phrases as they are."""


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


def repair_finding(finding: dict, counts: dict) -> None:
    """The per-finding repairs: blank quote to null, long quote cut, present implies mentioned, unquoted negation unmentioned."""
    evidence = finding.get("evidence")
    if isinstance(evidence, str) and evidence.strip() == "":
        evidence = None
        finding["evidence"] = None
    if isinstance(evidence, str) and len(evidence) > 200:
        finding["evidence"] = evidence[:200]
        counts["quote_truncated"] = counts.get("quote_truncated", 0) + 1
    if finding.get("present") is True and finding.get("mentioned") is not True:
        finding["mentioned"] = True
        counts["present_made_mentioned"] = counts.get("present_made_mentioned", 0) + 1
    if finding.get("mentioned") is True and finding.get("present") is not True and evidence is None:
        finding["mentioned"] = False
        counts["negation_without_quote_unmentioned"] = counts.get("negation_without_quote_unmentioned", 0) + 1
    if finding.get("mentioned") is False and finding.get("present") is not True and evidence is not None:
        finding["evidence"] = None
        counts["quote_dropped_for_unmentioned"] = counts.get("quote_dropped_for_unmentioned", 0) + 1


def normalize_finding(data: dict) -> tuple[dict, dict]:
    """Repairs for a single-field answer (question mode)."""
    counts = {}
    if isinstance(data, dict):
        repair_finding(data, counts)
    return data, counts


def sentence_around(quote: str, report_text: str) -> str:
    """The report sentence that contains the quote (whitespace and case folded), or "" when the quote is not found."""
    import re
    folded_report = re.sub(r"\s+", " ", report_text).lower()
    folded_quote = re.sub(r"\s+", " ", quote).strip().lower()
    position = folded_report.find(folded_quote) if folded_quote else -1
    if position == -1:
        return ""
    start = max(folded_report.rfind(". ", 0, position), folded_report.rfind("\n", 0, position), folded_report.rfind(": ", 0, position))
    end = folded_report.find(". ", position)
    return folded_report[start + 1 if start >= 0 else 0: end if end >= 0 else len(folded_report)]


def normalize(data: dict, report_text: str = "") -> tuple[dict, dict]:
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
        if isinstance(finding, dict):
            repair_finding(finding, counts)
    counts["quote_names_no_lobe"] = 0
    for name, phrases in QUOTE_MUST_CONTAIN.items():
        finding = data.get(name)
        if isinstance(finding, dict) and finding.get("present") is True:
            quote = (finding.get("evidence") or "").lower()
            context = quote + " " + sentence_around(quote, report_text)   # the quote itself, plus the sentence it was taken from
            if not all(any(re.search(r"\b" + re.escape(phrase) + r"\b", context) for phrase in group) for group in phrases):
                finding["mentioned"] = False
                finding["present"] = False
                finding["evidence"] = None
                counts["quote_names_no_lobe"] += 1
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


# ---- grouped extraction: several short calls per report instead of one 64-field call ----
def embolism_present(data: dict) -> bool:
    return isinstance(data.get("pulmonary_embolism"), dict) and data["pulmonary_embolism"].get("present") is True


def segmental_present(data: dict) -> bool:
    names = ["pe_segmental"] + [name for name in FINDING_NAMES if name.startswith("pe_segmental_")]
    return any(isinstance(data.get(name), dict) and data[name].get("present") is True for name in names)


CALL_GROUPS = [
    {"name": "core", "when": None, "fields": ["pulmonary_embolism", "pe_acute", "pe_chronic", "pe_saddle", "pe_multiple",
                                              "pe_occlusive", "pe_nonocclusive", "suboptimal_study", "poor_contrast_opacification", "motion_artifact"]},
    {"name": "levels", "when": embolism_present, "fields": ["pe_main", "pe_lobar", "pe_segmental", "pe_subsegmental", "pe_right", "pe_left",
                                                            "pe_main_right", "pe_main_left"]},
    {"name": "lobar", "when": embolism_present, "fields": [name for name in FINDING_NAMES if name.startswith("pe_lobar_")]},
    {"name": "segmental", "when": embolism_present, "fields": [name for name in FINDING_NAMES if name.startswith("pe_segmental_")]},
    {"name": "segments_right", "when": segmental_present, "fields": [name for name in FINDING_NAMES if name.startswith(("pe_segment_rul", "pe_segment_rml", "pe_segment_rll"))]},
    {"name": "segments_left", "when": segmental_present, "fields": [name for name in FINDING_NAMES if name.startswith(("pe_segment_lul", "pe_segment_lingula", "pe_segment_lll"))]},
    {"name": "other", "when": None, "fields": ["right_heart_strain", "pulmonary_artery_enlargement", "perfusion_defect", "pulmonary_infarct",
                                               "consolidation", "ground_glass_opacity", "atelectasis", "pulmonary_nodule", "emphysema",
                                               "mosaic_attenuation", "pleural_effusion", "pneumothorax", "pericardial_effusion", "cardiomegaly", "lymphadenopathy"]},
]
assert sorted(name for group in CALL_GROUPS for name in group["fields"]) == sorted(FINDING_NAMES)


# named segments of each lobe, for the fine mode: they are asked one by one only when that lobe's segmental field is present
SEGMENTS_BY_LOBE = {
    "pe_segmental_right_upper": ["pe_segment_rul_apical", "pe_segment_rul_anterior", "pe_segment_rul_posterior"],
    "pe_segmental_right_middle": ["pe_segment_rml_medial", "pe_segment_rml_lateral"],
    "pe_segmental_right_lower": ["pe_segment_rll_superior", "pe_segment_rll_medial_basal", "pe_segment_rll_anterior_basal", "pe_segment_rll_lateral_basal", "pe_segment_rll_posterior_basal"],
    "pe_segmental_left_upper": ["pe_segment_lul_apicoposterior", "pe_segment_lul_anterior"],
    "pe_segmental_lingula": ["pe_segment_lingula_superior", "pe_segment_lingula_inferior"],
    "pe_segmental_left_lower": ["pe_segment_lll_superior", "pe_segment_lll_anteromedial_basal", "pe_segment_lll_lateral_basal", "pe_segment_lll_posterior_basal"],
}
FINE_GROUPS = {"lobar", "segmental"}   # asked one field per call in fine mode; the named-segment groups are asked per lobe found


def fine_context(data: dict, field: str, group_fields: list) -> str:
    """What a single-field question needs to know to allocate a clot to the right level: the levels and arteries already
    found in this report, and the sibling fields that are answered in their own questions."""
    def state(name):
        finding = data.get(name) or {}
        return "present" if finding.get("present") else ("negated" if finding.get("mentioned") else "not mentioned")
    lines = ["Already extracted from this report, for orientation:"]
    lines.append("- levels: " + ", ".join(f"{name} {state(name)}" for name in ["pe_main", "pe_lobar", "pe_segmental", "pe_subsegmental"]))
    lines.append("- sides: " + ", ".join(f"{name} {state(name)}" for name in ["pe_right", "pe_left"]))
    found_lobar = [name for name in FINDING_NAMES if name.startswith("pe_lobar_") and (data.get(name) or {}).get("present")]
    found_segmental = [name for name in FINDING_NAMES if name.startswith("pe_segmental_") and (data.get(name) or {}).get("present")]
    if field.startswith("pe_segmental_"):
        lines.append("- lobar arteries found present: " + (", ".join(found_lobar) if found_lobar else "none") + ".")
        lines.append("Mistakes to avoid on this field, seen in earlier runs: (1) \"left lower lobe pulmonary artery\" or \"left upper and lower lobe arteries\" are LOBAR arteries, they do not make this field present; (2) \"subsegmental branches of both lower lobes\" or \"a subsegmental branch of the right lower lobe\" are SUBSEGMENTAL, they do not make this field present; (3) \"bilateral segmental branches\" or \"segmental branches\" without this lobe's name do not make this field present; (4) \"segmental arteries of the right middle lobe and right lower lobe\" names BOTH lobes, each of those two fields is present; (5) \"branches of all lobes\" makes every per-lobe segmental field present, this one included. Present only when the report says segmental artery or segmental branch(es) of this lobe, or \"both lower lobes\" / \"all lobes\" covering it.")
    if field.startswith("pe_segment_"):
        lines.append("- segmental involvement found by lobe: " + (", ".join(found_segmental) if found_segmental else "none") + ".")
        lines.append("Mistakes to avoid on this field, seen in earlier runs: (1) collective phrases (\"segmental branches\", \"multiple segmental and subsegmental branches\", \"all lobes\") never name a segment, answer not mentioned; (2) a named segment must not be dropped: \"anterior and lateral basal segmental arteries of the right lower lobe\" names TWO segments, anterior basal and lateral basal; \"a web in the posterior basal segmental branch\" names the posterior basal segment; (3) the segment word must match this field exactly (\"lateral basal\" is not \"lateral\" of the middle lobe; \"apicoposterior\" is a left upper lobe segment). Present only when this exact segment of this lobe is named.")
    if field.startswith("pe_lobar_"):
        lobe_words = FIELD_DEFINITIONS[field].split(" in the ")[1].split(" pulmonary")[0]
        lines.append("Mistakes to avoid on this field, seen in earlier runs: (1) \"" + lobe_words + " segmental arteries\" or \"segmental branches of the " + lobe_words + "\" are SEGMENTAL, not this field; (2) a lobe named only as the location of a segment (\"posterior basal segmental artery of the right lower lobe\") does not make the lobar artery involved; (3) \"extending into the right middle lobe and right lower lobe segmental arteries\" names segmental arteries, not the middle lobe artery; (4) the interlobar artery is its own field. Present only when the report names this lobe's pulmonary artery itself (\"" + lobe_words + " pulmonary artery\", \"" + lobe_words + " artery\", \"lower lobe arteries\" for a lower lobe) as containing clot.")
    siblings = [name for name in group_fields if name != field]
    lines.append("Answered in separate questions, do not put their findings here: " + "; ".join(f"{name} = {FIELD_DEFINITIONS[name]}" for name in siblings))
    return "\n".join(lines)


def group_model(group_name: str):
    """A Pydantic model with only that group's fields, for constrained decoding and validation."""
    from pydantic import create_model
    fields = next(group["fields"] for group in CALL_GROUPS if group["name"] == group_name)
    return create_model(f"Group_{group_name}", **{name: (Finding, ...) for name in fields})


def group_messages(report_text: str, group_name: str, include_schema: bool) -> list[dict]:
    fields = next(group["fields"] for group in CALL_GROUPS if group["name"] == group_name)
    system_prompt = EXTRACTION_PROMPT + "\n\nAnswer for these fields only, one JSON object with exactly these keys:\n" + "\n".join(
        f"- {name}: {FIELD_DEFINITIONS[name]}" for name in fields)
    if include_schema:
        system_prompt += "\n\nThe JSON object must match this JSON schema exactly:\n" + json.dumps(group_model(group_name).model_json_schema())
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Extract these findings from this report:\n\n" + report_text},
    ]


def normalize_group(data: dict) -> tuple[dict, dict]:
    """Per-finding repairs on one group's answer (the cross-field repairs run once all groups are merged)."""
    counts = {}
    if isinstance(data, dict):
        for finding in data.values():
            if isinstance(finding, dict):
                repair_finding(finding, counts)
    return data, counts
