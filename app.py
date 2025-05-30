import streamlit as st
import concurrent.futures
from PyPDF2 import PdfReader
import pandas as pd
import google.generativeai as genai
import openai
import json
import re
import os

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import letter

# --- Constants for Prompts and API ---
BA_SCALE_CONTENT = '''
# BEHAVIORAL ACTIVATION QUALITY SCALE¹

## Provider Information
- **Provider\'s Name:** _______________
- **Rater\'s Name:** _______________
- **Trial ID:** _______________

## Session Details
- **Date of Session:** _______________
- **Date of Rating:** _______________
- **Session #:** _______________
- **Phase #:** _______________

## Rating Method
- ☐ Audiotape = 1
- ☐ Live = 2
- ☐ Transcript = 3

## Supervision Type
- ☐ Group supervision = 1
- ☐ Individual supervision = 2

## Rating Source
- ☐ Self rating = 1
- ☐ Peer rating = 2
- ☐ Supervisor rating = 3
- ☐ Expert rating = 4

## Scoring Legend
- **0 = Not at all:** skill not performed
- **1 = Poor:** inappropriate performance with major problems evident; skill delivery is not useful in session
- **2 = Adequate:** skill performed adequately with some problems and/or inconsistencies
- **3 = Good:** Skill performed appropriately; minimal problems and/or consistencies; well-timed
- **4 = Excellent:** Skill is highly developed; helpful to the client even in the face of client difficulties; well-timed and consistently well-performed

---

## Intervention-Specific Skills

### 1. Uses the BA model (usesBAmodel)
**Description:** Explains the BA model in simple terms and checks that the client understands it. Asks questions such as: What happened? How did you feel? What did you do or not do? Also personalizes the BA model to the client\'s concerns, uses model to guide the selection of specific strategies, and checks that the client understands the BA model.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 2. Establishes and follows agenda (establishesAgenda)
**Description:** Works collaboratively with the client to plan a specific agenda relatively early in the session, focusing on behavioral activation, and follows the agenda during the session.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 3. Reviews and assigns HW (reviewsHW)
**Description:** Reviews and makes use of previously assigned homework with the client emphasizing learning and activation and develops one or more tasks for the client to engage in between sessions.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 4. Elicits commitment (elicitsCommitment)
**Description:** Obtains an agreement with the client to participate in intervention.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 5. Activity calendar and activity plan (Getting active) (activityCalendar)
**Description:** Uses the activity calendar and discusses mood ratings. Explains the connection between mood and activity. Plans activities that make the client feel good.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 6. Problem-solving (problemSolving)
**Description:** Introduces all the steps of problem solving: 1) defining the problem, 2) generating solutions and 3) selecting appropriate solutions in collaboration with the client.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 7. Strategies for specific problems (strategiesForSpecificProblems)
**Description:** Uses strategies like relaxation exercises that will help the client with specific problems.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 8. Manages barriers during the session (managesBarriers)
**Description:** Deals with any challenges that arise during the session (e.g., lack of privacy and interruptions from spouse or child)

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 9. Involves a significant other (involvesSignificantOther)
**Description:** Asks if the client wants to involve a significant other and how he/she would like to do that.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 10. Suicide risk assessment (suicideRiskAssessment)
**Description:** Assesses the degree of suicide risk and takes appropriate action based on the assessment.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

**Total Score:** _______________
**Mean Score:** Total Number/(10-N/As) = _______________

---

## General Skills

### 1. Rapport-building & self-disclosure (rapportBuilding)
**Description:** Makes casual informal conversation and shares relatable experiences with the client. Introduces self and role, asks the client to introduce themselves, and elicits their reason for accessing the intervention.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 2. Confidentiality (confidentiality)
**Description:** Explains confidentiality, discusses limitations of confidentiality (and why these exist), checks client\'s understanding about topics discussed.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 3. Active listening (activeListening)
**Description:** Listens attentively through non-verbal behavior (eye contact, nodding, open body posture) and verbal behavior (e.g., "mhm").

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 4. Open-ended questions & reflections (openEndedQuestions)
**Description:** Uses open-ended questions (beyond yes/no responses) and mirrors back the client\'s feelings to convey understanding of what he/she says.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 5. Empathy, warmth, and authenticity (empathy)
**Description:** Demonstrates accurate understanding, acknowledges client\'s experience, displays warmth, and appears natural and genuine in interactions. Does not try to sound like somebody else or put their needs above those of the client.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 6. Collaborative (collaborative)
**Description:** Checks in with the client, frequently, about their understanding while planning intervention related activities. Creates opportunities for the client to actively participate in the session.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 7. Validates client\'s experience (validatesExperience)
**Description:** Shows that he/she understands the client\'s experience and communicates that these experiences make sense within context.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 8. Encouraging (Promoting realistic change for hope) (encouraging)
**Description:** Encourages the client\'s progress even in the face of obstacles and promotes realistic hope for change.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 9. Elicits affect (elicitsAffect)
**Description:** Appropriately encourages the client to share feelings, explains that others may share similar feelings in similar situations. Asks the client to reflect on the experience of sharing emotions.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

### 10. Summarizes (summarizes)
**Description:** Highlights what has been said, shows that they have been listening to the client carefully, and prepares the client to move on.

**Rating:** ☐ 0 Not Done ☐ 1 Poor ☐ 2 Adequate ☐ 3 Good ☐ 4 Excellent ☐ N/A Not Applicable

**Total Score:** _______________
**Mean Score:** Total Number/(10-N/As) = _______________

---

## Overall Assessment

**Total Mean Score:** _______________ + _______________ = _______________
(Mean Intervention-specific) + (Mean General) = (Total Score)

### Additional Ratings

**How would you rate the difficulty level of working with this client?**
☐ 0 Not Difficult ☐ 1 ☐ 2 Moderately Difficult ☐ 3 ☐ 4 Extremely Difficult

**How would you rate the quality of the audiotape?**
☐ 0 Poor ☐ 1 ☐ 2 Adequate ☐ 3 ☐ 4 Excellent

**Overall, how would you rate the provider?**
☐ 0 Insufficient ☐ 1 ☐ 2 Adequate ☐ 3 ☐ 4 Excellent

**Total Score:** _______________
**Mean Score:** Total Number/(3) = _______________

---

## Comments
*Include red flags, if any:*

_______________________________________________________________________________
_______________________________________________________________________________
_______________________________________________________________________________

---

¹ *Adapted from: Singla, D. R., Weobong, B., Nadkarni, A., Chowdhary, N., Shinde, S., Anand, A., Fairburn, C. G., Dimijdan, S., Velleman, R., Weiss, H., & Patel, V. (2014). Improving the scalability of psychological treatments in developing countries: an evaluation of peer-led therapy* 
'''

SCHEMA_EXAMPLE_CONTENT = '''
Create a structured output using the Behavioral Activation (BA) quality scale, ensuring that item ratings are nested under two sections: intervention-specific skills and general skills.

Include the following fields for each section:
- Provider\'s Name
- Rater\'s Name
- Trial ID
- Date of Session
- Date of Rating
- Session Number
- Phase Number
- Scoring Legend: 0 (Not at all) to 4 (Excellent)

# Steps

1. For each item under the intervention-specific skills, enter the rating based on the criteria provided.
2. For each item under the general skills, enter the rating based on the criteria provided.
3. Calculate the total and mean scores for each section separately.
4. Calculate the total mean score by averaging the mean scores of intervention-specific skills and general skills.
5. Rate additional aspects such as the difficulty level of working with the client, the quality of the audiotape, and the overall provider quality.

# Output Format

```json
{
  "providerName": "Provider\'s Name",
  "raterName": "Rater\'s Name",
  "trialId": "Trial ID",
  "sessionDate": "Date of Session",
  "ratingDate": "Date of Rating",
  "sessionNumber": "Session #",
  "phaseNumber": "Phase #",
  "interventionSpecificSkills": {
    "usesBAmodel": "Rating",
    "establishesAgenda": "Rating",
    "reviewsHW": "Rating",
    "elicitsCommitment": "Rating",
    "activityCalendar": "Rating",
    "problemSolving": "Rating",
    "strategiesForSpecificProblems": "Rating",
    "managesBarriers": "Rating",
    "involvesSignificantOther": "Rating",
    "suicideRiskAssessment": "Rating",
    "totalScore": "Total Score",
    "meanScore": "Mean Score"
  },
  "generalSkills": {
    "rapportBuilding": "Rating",
    "confidentiality": "Rating",
    "activeListening": "Rating",
    "openEndedQuestions": "Rating",
    "empathy": "Rating",
    "collaborative": "Rating",
    "validatesExperience": "Rating",
    "encouraging": "Rating",
    "elicitsAffect": "Rating",
    "summarizes": "Rating",
    "totalScore": "Total Score",
    "meanScore": "Mean Score"
  },
  "totalMeanScore": "Total Mean Score",
  "difficultyLevel": "Rating",
  "audiotapeQuality": "Rating",
  "providerOverallRating": "Rating"
}
```

# Notes

- Ensure that each rating is based on the specific criteria provided for each skill or aspect.
- Use \'N/A\' where an item is not applicable.
- The total scores should reflect the sum of all rated items within their respective skills section.
- Mean scores are calculated by dividing the total score by the number of applicable items evaluated. 
'''

# --- Model Names (Global Constants) ---
GEMINI_DETAILED_SUMMARY_MODEL_NAME = 'gemini-1.5-pro-latest'
GEMINI_RATINGS_MODEL_NAME = 'gemini-2.5-pro-preview-05-06' # Specified in api-examples.md
OPENAI_CONCISE_SUMMARY_MODEL_NAME = 'gpt-4o' # As per project plan

# --- Utility Functions ---
def extract_text_from_pdf(uploaded_file_object):
    """Extracts text from all pages of an uploaded PDF file object,
    performing cleaning to remove problematic characters.
    """
    text_parts = []
    try:
        uploaded_file_object.seek(0)
        pdf_reader = PdfReader(uploaded_file_object)
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    cleaned_text = re.sub(r'[\t ]+', ' ', page_text)
                    cleaned_text = re.sub(r'(\r\n|\r|\n)', '\n', cleaned_text)
                    cleaned_text = re.sub(r'[^\x20-\x7E\n\r\t]+', '', cleaned_text)
                    replacements = {
                        '“': '"', '”': '"',
                        '‘': "'", '’': "'",
                        '—': '-',
                        '–': '-',
                    }
                    for char_old, char_new in replacements.items():
                        cleaned_text = cleaned_text.replace(char_old, char_new)
                    text_parts.append(cleaned_text)
            except Exception as page_e:
                st.warning(f"Could not process page {page_num + 1} in {uploaded_file_object.name}: {page_e}")
                continue
    except Exception as e:
        st.error(f"Error reading or processing PDF {uploaded_file_object.name}: {e}")
        return None
    return "\n".join(text_parts).strip() if text_parts else None

def get_gemini_detailed_summary(api_key, transcript_text):
    """Gets a detailed summary from Gemini.
    Uses the model name 'gemini-2.5-pro-preview-05-06'.
    """
    if not api_key:
        st.error("Gemini API Key is not provided.")
        return None
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06')
    prompt = (
        "Provide a detailed summary of the transcription text, specifically about each scenario, "
        "the clinician's performance and any relevant feedback they received from the trainer."
    )
    full_prompt = f"{prompt}\n\nTranscription Text:\n{transcript_text}"
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API for detailed summary: {e}")
        return None

def get_openai_concise_summary(api_key, detailed_summary_text):
    """Gets a concise summary from OpenAI's gpt-4o model."""
    if not api_key:
        st.error("OpenAI API Key is not provided.")
        return None
    if not detailed_summary_text:
        st.warning("Detailed summary (from Gemini) is not available for OpenAI processing.")
        return None
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in summarizing clinical session transcripts."},
                {"role": "user", "content": f"Condense the following detailed summary into a concise overview capturing all of its salient points. The output should be a single paragraph.\n\nDetailed Summary:\n{detailed_summary_text}"}
            ],
            temperature=0.7, # Adjusted for a balance of creativity and factuality
            max_tokens=500  # Allowing for a reasonably sized concise summary
        )
        concise_summary = response.choices[0].message.content.strip()
        return concise_summary
    except Exception as e:
        st.error(f"Error calling OpenAI API for concise summary: {e}")
        return None

def get_gemini_ratings_and_justifications(api_key, detailed_summary):
    """Gets ratings and justifications from Gemini using a predefined schema and context.
    Uses the model name 'gemini-2.5-pro-preview-05-06'.
    Relies on global BA_SCALE_CONTENT and SCHEMA_EXAMPLE_CONTENT constants.
    """
    if not api_key:
        st.error("Gemini API Key is not provided for ratings.")
        return None
    if not detailed_summary:
        st.warning("Detailed summary is not available for ratings.")
        return None

    api_response_schema = {
        "type": "OBJECT",
        "properties": {
            "providerName": {"type": "STRING"},
            "raterName": {"type": "STRING"},
            "trialId": {"type": "STRING"},
            "sessionDate": {"type": "STRING"},
            "ratingDate": {"type": "STRING"},
            "sessionNumber": {"type": "STRING"},
            "phaseNumber": {"type": "STRING"},
            "interventionSpecificSkills": {
                "type": "OBJECT",
                "properties": {
                    **{skill: {
                        "type": "OBJECT",
                        "properties": {
                            "rating": {"type": "STRING"},
                            "justification": {"type": "STRING"}
                        },
                        "required": ["rating", "justification"]
                    } for skill in [
                        "usesBAmodel", "establishesAgenda", "reviewsHW", "elicitsCommitment",
                        "activityCalendar", "problemSolving", "strategiesForSpecificProblems",
                        "managesBarriers", "involvesSignificantOther", "suicideRiskAssessment"
                    ]},
                    "totalScore": {"type": "STRING"},
                    "meanScore": {"type": "STRING"}
                },
                "required": [
                    "usesBAmodel", "establishesAgenda", "reviewsHW", "elicitsCommitment",
                    "activityCalendar", "problemSolving", "strategiesForSpecificProblems",
                    "managesBarriers", "involvesSignificantOther", "suicideRiskAssessment",
                    "totalScore", "meanScore"
                ]
            },
            "generalSkills": {
                "type": "OBJECT",
                "properties": {
                     **{skill: {
                        "type": "OBJECT",
                        "properties": {
                            "rating": {"type": "STRING"},
                            "justification": {"type": "STRING"}
                        },
                        "required": ["rating", "justification"]
                    } for skill in [
                        "rapportBuilding", "confidentiality", "activeListening", "openEndedQuestions",
                        "empathy", "collaborative", "validatesExperience", "encouraging",
                        "elicitsAffect", "summarizes"
                    ]},
                    "totalScore": {"type": "STRING"},
                    "meanScore": {"type": "STRING"}
                },
                "required": [
                    "rapportBuilding", "confidentiality", "activeListening", "openEndedQuestions",
                    "empathy", "collaborative", "validatesExperience", "encouraging",
                    "elicitsAffect", "summarizes", 
                    "totalScore", "meanScore"
                ]
            },
            "totalMeanScore": {"type": "STRING"},
            "difficultyLevel": {"type": "STRING"},
            "audiotapeQuality": {"type": "STRING"},
            "providerOverallRating": {"type": "STRING"}
        },
        "required": [
            "providerName", "raterName", "trialId", "sessionDate", "ratingDate", 
            "sessionNumber", "phaseNumber", "interventionSpecificSkills", "generalSkills", 
            "totalMeanScore", "difficultyLevel", "audiotapeQuality", "providerOverallRating"
        ]
    }

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-pro-preview-05-06')

    prompt = f"""
    You are an expert in evaluating clinical skills based on transcripts. Please analyze the provided detailed summary of a therapy session and the Behavioral Activation (BA) scale definitions. Your goal is to fill out a structured JSON report according to the schema I will provide to the API.

    DETAILED SUMMARY:
    {detailed_summary}

    BEHAVIORAL ACTIVATION (BA) SCALE DEFINITIONS (includes item names and descriptions):
    {BA_SCALE_CONTENT}

    DESIRED OUTPUT STRUCTURE EXAMPLE (for your understanding, the actual schema is enforced by the API and has specific field names):
    {SCHEMA_EXAMPLE_CONTENT}

    INSTRUCTIONS:
    1.  For each skill item (e.g., 'usesBAmodel', 'rapportBuilding') listed in the BA scale definitions, provide two pieces of information:
        a.  'rating': A string representing the score. The format should be "Number DescriptiveWord" (e.g., "4 Excellent", "3 Good", "2 Adequate", "1 Poor", "0 Not at all"). If not applicable, use "N/A". Refer to the BA scale's scoring legend.
        b.  'justification': A concise, single-sentence justification for your rating, directly referencing evidence from the DETAILED SUMMARY. If a skill is rated "N/A" or "0 Not at all" because it wasn't demonstrated or applicable, state that in the justification.
    2.  The skill item keys in your JSON output (e.g., "usesBAmodel") MUST EXACTLY MATCH the keys defined in the API schema (which correspond to the camelCase versions of the skills in the BA Scale Definitions).
    3.  Populate all top-level fields like 'providerName', 'raterName', 'trialId', etc. Use "Not Specified" as a string if the information is not present in the summary. For fields like 'ratingDate', 'sessionNumber', try to infer them or use a sensible placeholder like a date or number if appropriate, otherwise use "Not Specified".
    4.  For 'totalScore' and 'meanScore', these should be direct string properties of the 'interventionSpecificSkills' and 'generalSkills' objects respectively. Calculate these based on your ratings for those sections. Ensure these are also strings. If all items in a section are "N/A", the scores can also be "N/A".
    5.  Format 'audiotapeQuality' and 'providerOverallRating' similarly to skill ratings (e.g., "4 Excellent", "3 Good", "N/A").
    6.  Your entire response MUST be a single, valid JSON object that strictly adheres to the schema provided to the API.
    """

    try:
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=api_response_schema
        )
        
        st.info("Sending request to Gemini for ratings with schema...")
        response = model.generate_content(
            prompt, 
            generation_config=generation_config,
        )
        
        st.success("Received response from Gemini for ratings.")
        
        response_text = ""
        if response.parts:
            first_part = response.parts[0]
            if hasattr(first_part, 'text'):
                response_text = first_part.text
            elif hasattr(first_part, 'json_data'): 
                 return first_part.json_data 
        elif hasattr(response, 'text'):
            response_text = response.text

        if not response_text:
             st.error("Gemini API returned an empty response for ratings.")
             return None

        return json.loads(response_text)

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON response from Gemini: {e}")
        error_text_to_display = response_text if 'response_text' in locals() and response_text else "No response text available or response object was malformed."
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
            error_text_to_display += f"\nPrompt Feedback: {response.prompt_feedback}"
        st.text_area("Problematic Gemini Response Text:", error_text_to_display, height=200)
        return None
    except Exception as e:
        st.error(f"Error calling Gemini API for ratings with schema: {e}")
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
            st.error(f"Prompt Feedback: {response.prompt_feedback}")
        return None

def parse_ba_scale_to_dict(ba_scale_md_content):
    """Parses the BA scale markdown content (now from an internal constant) into a dictionary.
    Maps skill_key to {item_name, description}.
    Relies on skill keys being explicitly defined in parentheses in the headings.
    """
    skills_dict = {}
    if not ba_scale_md_content:
        st.error("Internal BA Scale content is empty, cannot parse. This is an application error.")
        return skills_dict

    pattern = re.compile(
        r"### \\d+\\. (.*?) \\((.*?)\\)\\n"  # Group 1: Item Name, Group 2: Skill Key
        r"\\s*?"
        r"\\*\\*Description:\\*\\* (.*?)"
        r"(?=\\n\\s*\\*\\*Rating:\\*\\*)",
        re.DOTALL
    )

    matches = pattern.findall(ba_scale_md_content)
    for match in matches:
        item_name = match[0].strip()
        skill_key = match[1].strip()
        description_text = match[2].strip().replace('\n', ' ')
        
        if skill_key and item_name: # Description can be empty but key and name must exist
            skills_dict[skill_key] = {
                "item_name": item_name,
                "description": description_text
            }
        # else:
            # Consider if a warning is needed for development/debugging if a block is malformed
            # st.warning(f"Could not fully parse a skill block from internal BA Scale. Title: '{item_name}', Key: '{skill_key}'")

    if not skills_dict:
        # This warning is for a failure to parse the *internal* constant, which is an app issue.
        st.warning("Could not parse any skills from the internal BA Scale definitions. PDF reports may lack detail.")
        # REMOVE: st.text_area("BA Scale Content for Regex Debug:", ba_scale_md_content, height=200)
        # REMOVE: st.text_area("Regex Used:", pattern.pattern, height=70)
    # else:
        # REMOVE: st.success(f"Successfully parsed {len(skills_dict)} skills from ba-scale.md.")

    return skills_dict

def prepare_data_for_pdf_tables(ratings_justifications, ba_scale_dict):
    """Prepares data into a format suitable for PDF table generation and extracts metadata."""
    prepared_data = {
        "intervention_specific_skills_table": [],
        "general_skills_table": [],
        "intervention_specific_total_score": None,
        "intervention_specific_mean_score": None,
        "general_skills_total_score": None,
        "general_skills_mean_score": None,
        "overall_total_mean_score": None,
        "audiotape_quality": None,
        "difficulty_level": None,
        "provider_overall_rating": None,
        "provider_name": None,
        "rater_name": None,
        "trial_id": None,
        "session_date": None,
        "rating_date": None,
        "session_number": None,
        "phase_number": None
    }

    if not ratings_justifications or not ba_scale_dict:
        st.error("Missing ratings/justifications or ba_scale dictionary for preparing table data.")
        return prepared_data

    # Extract top-level metadata
    prepared_data["overall_total_mean_score"] = ratings_justifications.get("totalMeanScore")
    prepared_data["audiotape_quality"] = ratings_justifications.get("audiotapeQuality")
    prepared_data["difficulty_level"] = ratings_justifications.get("difficultyLevel")
    prepared_data["provider_overall_rating"] = ratings_justifications.get("providerOverallRating")
    prepared_data["provider_name"] = ratings_justifications.get("providerName")
    prepared_data["rater_name"] = ratings_justifications.get("raterName")
    prepared_data["trial_id"] = ratings_justifications.get("trialId")
    prepared_data["session_date"] = ratings_justifications.get("sessionDate")
    prepared_data["rating_date"] = ratings_justifications.get("ratingDate")
    prepared_data["session_number"] = ratings_justifications.get("sessionNumber")
    prepared_data["phase_number"] = ratings_justifications.get("phaseNumber")

    skill_groups_map = {
        "interventionSpecificSkills": "intervention_specific_skills_table",
        "generalSkills": "general_skills_table"
    }

    for group_key, table_key in skill_groups_map.items():
        skill_group_data = ratings_justifications.get(group_key, {})
        if group_key == "interventionSpecificSkills":
            prepared_data["intervention_specific_total_score"] = skill_group_data.get("totalScore")
            prepared_data["intervention_specific_mean_score"] = skill_group_data.get("meanScore")
        elif group_key == "generalSkills":
            prepared_data["general_skills_total_score"] = skill_group_data.get("totalScore")
            prepared_data["general_skills_mean_score"] = skill_group_data.get("meanScore")

        for skill_id, skill_ratings in skill_group_data.items():
            if skill_id in ["totalScore", "meanScore"]: # Skip summary scores within skills list
                continue
            
            ba_info = ba_scale_dict.get(skill_id)
            if ba_info:
                row = {
                    "Item": ba_info.get("item_name", skill_id), # Fallback to skill_id if item_name missing
                    "Description": ba_info.get("description", "N/A"),
                    "Rating": skill_ratings.get("rating", "N/A"),
                    "Justification": skill_ratings.get("justification", "N/A")
                }
                prepared_data[table_key].append(row)
            else:
                # Handle skills present in ratings but not in ba_scale.md (e.g. if schema changes)
                st.warning(f"Skill ID '{skill_id}' found in ratings but not in parsed ba-scale.md. Using placeholder info.")
                row = {
                    "Item": skill_id,
                    "Description": "Description not found in ba-scale.md",
                    "Rating": skill_ratings.get("rating", "N/A"),
                    "Justification": skill_ratings.get("justification", "N/A")
                }
                prepared_data[table_key].append(row)

    return prepared_data

def ensure_output_directory_exists():
    """Ensures the 'outputs' directory exists."""
    if not os.path.exists("outputs"):
        try:
            os.makedirs("outputs")
            st.info("Created 'outputs' directory for PDF reports.")
        except Exception as e:
            st.error(f"Could not create 'outputs' directory: {e}")
            return False
    return True

def _create_skill_table(skill_data_list, styles, col_widths):
    """Helper function to create a ReportLab Table object for skills."""
    if not skill_data_list:
        return Paragraph("[No data available for this skill section]")

    headers = ["Item", "Description", "Rating", "Justification"]
    table_data = [headers]   
    for row_data in skill_data_list:
        # Wrap text in Paragraphs for better flow and styling within cells
        item_p = Paragraph(str(row_data.get("Item", "N/A")), styles['Normal'])
        desc_p = Paragraph(str(row_data.get("Description", "N/A")), styles['Normal'])
        rating_p = Paragraph(str(row_data.get("Rating", "N/A")), styles['Normal'])
        just_p = Paragraph(str(row_data.get("Justification", "N/A")), styles['Normal'])
        table_data.append([item_p, desc_p, rating_p, just_p])
    
    table = Table(table_data, colWidths=col_widths)
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'TOP'), # Align text to top of cell
    ])
    table.setStyle(table_style)
    return table

def generate_pdf_report(file_data, original_filename_base, include_appendix):
    """Generates a PDF report using ReportLab."""
    if not ensure_output_directory_exists():
        return None

    output_pdf_path = os.path.join("outputs", f"{original_filename_base}_report.pdf")
    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch, leftMargin=0.75*inch, rightMargin=0.75*inch)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles (ensure they are defined or adjust as needed)
    title_style = ParagraphStyle('TitleCustom', parent=styles['h1'], alignment=TA_CENTER, spaceAfter=0.2*inch, fontSize=16)
    heading_style = ParagraphStyle('HeadingCustom', parent=styles['h2'], spaceBefore=0.2*inch, spaceAfter=0.1*inch, fontSize=14)
    body_style = ParagraphStyle('BodyCustom', parent=styles['BodyText'], alignment=TA_JUSTIFY, spaceBefore=6, spaceAfter=6, leading=14)
    metadata_style = ParagraphStyle('Metadata', parent=styles['Normal'], spaceAfter=0.05*inch, leading=14, fontSize=10)
    score_style = ParagraphStyle('Score', parent=styles['Normal'], spaceBefore=0.1*inch, leading=14, fontSize=10, fontWeight='Bold')

    # Report Title
    story.append(Paragraph(f"Behavioral Activation Adherence Report: {original_filename_base}", title_style))
    story.append(Spacer(1, 0.1*inch))

    # Concise Summary
    concise_summary = file_data.get("summary", "Summary not available.")
    story.append(Paragraph("Session Summary", heading_style))
    story.append(Paragraph(concise_summary, body_style))
    story.append(Spacer(1, 0.15*inch))

    # Key Information Section (Metadata)
    story.append(Paragraph("Key Session Information", heading_style))
    pdf_table_data = file_data.get("pdf_table_data")
    if pdf_table_data:
        meta_data_to_display = [
            (f"Provider Name: {pdf_table_data.get('provider_name', 'N/A')}", f"Session Number: {pdf_table_data.get('session_number', 'N/A')}"),
            (f"Rater Name: {pdf_table_data.get('rater_name', 'N/A')}", f"Phase Number: {pdf_table_data.get('phase_number', 'N/A')}"),
            (f"Trial ID: {pdf_table_data.get('trial_id', 'N/A')}", f"Audiotape Quality: {pdf_table_data.get('audiotape_quality', 'N/A')}"),
            (f"Session Date: {pdf_table_data.get('session_date', 'N/A')}", f"Difficulty Level: {pdf_table_data.get('difficulty_level', 'N/A')}"),
            (f"Rating Date: {pdf_table_data.get('rating_date', 'N/A')}", f"Provider Overall Rating: {pdf_table_data.get('provider_overall_rating', 'N/A')}")
        ]
        
        # Create a 2-column table for metadata for better layout
        meta_table_data = [[Paragraph(item[0], metadata_style), Paragraph(item[1], metadata_style)] for item in meta_data_to_display]
        meta_table = Table(meta_table_data, colWidths=[3*inch, 3*inch]) # Adjust colWidths as needed
        meta_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
        story.append(meta_table)
    else:
        story.append(Paragraph("Key session information not available.", body_style))
    story.append(Spacer(1, 0.15*inch))

    # Column widths for skill tables (adjust as needed - total should be less than page width minus margins)
    # Page width (letter) is 8.5 inches. Margins L+R = 0.75+0.75 = 1.5 inches. Available width = 7 inches.
    col_widths = [1.5*inch, 2.5*inch, 0.75*inch, 2.25*inch] 

    # Intervention Specific Skills Table
    story.append(Paragraph("Intervention-Specific Skills", heading_style))
    if pdf_table_data and pdf_table_data.get("intervention_specific_skills_table"):
        iss_table = _create_skill_table(pdf_table_data["intervention_specific_skills_table"], styles, col_widths)
        story.append(iss_table)
        story.append(Spacer(1, 0.05*inch))
        story.append(Paragraph(f"Total Score: {pdf_table_data.get('intervention_specific_total_score', 'N/A')}", score_style))
        story.append(Paragraph(f"Mean Score: {pdf_table_data.get('intervention_specific_mean_score', 'N/A')}", score_style))
    else:
        story.append(Paragraph("[Intervention-Specific Skills data not available.]", body_style))
    story.append(Spacer(1, 0.15*inch))

    # General Skills Table
    story.append(Paragraph("General Skills", heading_style))
    if pdf_table_data and pdf_table_data.get("general_skills_table"):
        gs_table = _create_skill_table(pdf_table_data["general_skills_table"], styles, col_widths)
        story.append(gs_table)
        story.append(Spacer(1, 0.05*inch))
        story.append(Paragraph(f"Total Score: {pdf_table_data.get('general_skills_total_score', 'N/A')}", score_style))
        story.append(Paragraph(f"Mean Score: {pdf_table_data.get('general_skills_mean_score', 'N/A')}", score_style))
    else:
        story.append(Paragraph("[General Skills data not available.]", body_style))
    story.append(Spacer(1, 0.15*inch))

    # Overall Assessment Section (already has Total Mean Score)
    story.append(Paragraph("Overall Assessment Summary", heading_style))
    if pdf_table_data:
        story.append(Paragraph(f"Provider Overall Rating: {pdf_table_data.get('provider_overall_rating', 'N/A')}", score_style)) # Duplicating here from metadata for emphasis
        story.append(Paragraph(f"Total Mean Score (All Skills): {pdf_table_data.get('overall_total_mean_score', 'N/A')}", score_style))
    else:
        story.append(Paragraph("Overall assessment data not available.", body_style))
    story.append(Spacer(1, 0.2*inch))

    # Conditionally add Detailed Summary Appendix
    if include_appendix:
        detailed_summary_text = file_data.get("detailed_summary", "Detailed summary not available for appendix.")
        if detailed_summary_text != "Detailed summary not available for appendix.": # Check if it's the actual summary
            story.append(PageBreak())
            story.append(Paragraph("Appendix: Detailed Session Summary", title_style)) # Re-use title_style or a specific appendix_title_style
            story.append(Spacer(1, 0.1*inch))
            # Split detailed summary into paragraphs to handle long text better
            detailed_summary_paragraphs = detailed_summary_text.split('\n')
            for para_text in detailed_summary_paragraphs:
                if para_text.strip(): # Add non-empty paragraphs
                    story.append(Paragraph(para_text, body_style))
                    story.append(Spacer(1, 0.05*inch)) # Small spacer between paragraphs
    
    try:
        doc.build(story)
        st.success(f"Successfully generated PDF report: {output_pdf_path}")
        return output_pdf_path
    except Exception as e:
        st.error(f"Error generating PDF report for {original_filename_base}: {e}")
        return None

st.set_page_config(layout="wide")

# Sidebar for API keys and settings
st.sidebar.title("API Credentials & Settings")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Gemini API Key.")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API Key.")
processing_timeout = st.sidebar.number_input("Processing Timeout (seconds)", min_value=30, max_value=600, value=300, step=10, help="Set the timeout for API calls in seconds.")
retry_count = st.sidebar.number_input("Retry Count", min_value=0, max_value=5, value=1, step=1, help="Set the number of retries for failed API calls.")
include_detailed_summary_appendix = st.sidebar.checkbox("Include Detailed Summary as Appendix in PDF", value=False)

# Main area
st.title("PSii Transcription Review")
st.write("Upload your PDF files to begin processing.")

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Drag and drop or browse to upload multiple PDF files for transcription review."
)

if uploaded_files:
    st.write(f"{len(uploaded_files)} file(s) uploaded:")
    for uploaded_file in uploaded_files:
        st.write(uploaded_file.name)

    if st.button("Start Processing"):
        if not gemini_api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
        else:
            st.info("Processing started...")
            
            results = [] 

            # Use the global BA_SCALE_CONTENT for parsing BA scale definitions
            ba_scale_dictionary = parse_ba_scale_to_dict(BA_SCALE_CONTENT)
            if not ba_scale_dictionary:
                st.error("Failed to parse embedded BA Scale content. Critical for table generation. Processing aborted for all files.")
                # return # or st.stop() if preferred to halt execution
            
            for uploaded_file in uploaded_files:
                st.write(f"Processing {uploaded_file.name}...")
                extracted_text = extract_text_from_pdf(uploaded_file)
                if extracted_text:
                    detailed_summary = get_gemini_detailed_summary(gemini_api_key, extracted_text)
                    if detailed_summary:
                        st.subheader(f"Detailed Summary for {uploaded_file.name}:")
                        st.markdown(detailed_summary)
                        current_file_result = {
                            "file": uploaded_file.name, 
                            "detailed_summary": detailed_summary,
                            "summary": detailed_summary, 
                            "ratings": None, 
                            "pdf_table_data": None, 
                            "report_pdf_path": None
                        }

                        concise_summary = None
                        if openai_api_key: 
                            concise_summary = get_openai_concise_summary(openai_api_key, detailed_summary)
                            if concise_summary:
                                st.subheader(f"Concise Summary for {uploaded_file.name}:")
                                st.markdown(concise_summary)
                                current_file_result["summary"] = concise_summary
                            else:
                                st.warning(f"Could not generate concise summary for {uploaded_file.name}. Using detailed summary in main report.")
                        else:
                            st.warning("OpenAI API Key not provided. Skipping concise summary generation. Detailed summary will be used in main report.")
                        
                        results.append(current_file_result) 
                        
                        ratings_justifications = get_gemini_ratings_and_justifications(
                            gemini_api_key,
                            current_file_result["summary"]
                        )
                        if ratings_justifications:
                            st.subheader(f"Ratings & Justifications (JSON) for {uploaded_file.name}:")
                            st.json(ratings_justifications) 
                            current_file_result["ratings"] = ratings_justifications

                            # Step 11: Assemble table data
                            if ba_scale_dictionary: # Ensure ba_scale_dictionary was successfully parsed
                                pdf_table_data = prepare_data_for_pdf_tables(ratings_justifications, ba_scale_dictionary)
                                current_file_result["pdf_table_data"] = pdf_table_data
                                st.subheader(f"Prepared Data for PDF Tables (File: {uploaded_file.name})")
                                st.json(pdf_table_data) # Displaying for verification

                                # Step 12: Generate PDF Report
                                original_fname_base, _ = os.path.splitext(uploaded_file.name)
                                report_path = generate_pdf_report(current_file_result, original_fname_base, include_detailed_summary_appendix) # Pass checkbox state
                                if report_path:
                                    current_file_result["report_pdf_path"] = report_path
                                    with open(report_path, "rb") as pdf_file:
                                        st.download_button(
                                            label=f"Download Report for {uploaded_file.name}",
                                            data=pdf_file,
                                            file_name=os.path.basename(report_path),
                                            mime="application/pdf"
                                        )
                            else:
                                st.warning(f"Skipping PDF table data preparation for {uploaded_file.name} due to ba-scale.md parsing issues.")
                        else:
                            st.warning(f"Could not generate ratings/justifications for {uploaded_file.name}.")
                    else:
                        st.warning(f"Could not generate summary for {uploaded_file.name}.")
                else:
                    st.warning(f"Could not extract text from {uploaded_file.name}.")
            
            if any(r.get("summary") for r in results): # Check if any summary was generated
                st.success("Processing complete for all uploaded files!")
            else:
                st.warning("Processing finished, but no summaries were generated.")
else:
    st.info("Please upload PDF files to begin.")

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# You can access the values like this:
# if gemini_api_key:
#     st.sidebar.success("Gemini API Key loaded.")
# if openai_api_key:
#     st.sidebar.success("OpenAI API Key loaded.")
# st.sidebar.write(f"Timeout: {processing_timeout}s, Retries: {retry_count}") 