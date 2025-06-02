import streamlit as st
import concurrent.futures
from PyPDF2 import PdfReader
import pandas as pd
import google.generativeai as genai
import openai
import json
import re
import os
import time # Added for retry delay

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import letter

# Moved UI update function to the top
def update_progress_ui(progress_bar, status_placeholder, file_statuses, total_files):
    processed_count = 0
    for filename, statuses in file_statuses.items():
        if statuses.get("pdf_report_status") == "Complete" or \
           statuses.get("pdf_report_status") == "Error" or \
           statuses.get("pdf_report_status") == "Skipped":
            processed_count += 1

    progress_value = 0
    if total_files > 0:
        progress_value = processed_count / total_files
    progress_bar.progress(progress_value)
    
    status_texts = []
    for filename, statuses in file_statuses.items():
        text = f"<b>{filename}:</b> "
        text += f"Summary: {statuses.get('detailed_summary_status', 'Pending')}, "
        text += f"Ratings: {statuses.get('ratings_status', 'Pending')}, "
        text += f"Concise: {statuses.get('concise_summary_status', 'Pending')}, "
        text += f"Report: {statuses.get('pdf_report_status', 'Pending')}"
        status_texts.append(text)
    status_placeholder.markdown("\n".join(status_texts), unsafe_allow_html=True)

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

def get_gemini_detailed_summary(api_key, transcript_text, timeout, retries):
    """Gets a detailed summary from Gemini.
    Uses the model name 'gemini-2.5-pro-preview-05-06'.
    Includes timeout and retry logic.
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
    
    for attempt in range(retries + 1):
        try:
            response = model.generate_content(
                full_prompt,
                request_options={
                    "timeout": timeout
                }
            )
            return response.text
        except Exception as e:
            st.warning(f"Gemini API (summary) attempt {attempt + 1} of {retries + 1} failed: {e}")
            if attempt < retries:
                time.sleep(1)  # Wait 1 second before retrying
            else:
                st.error(f"Gemini API (summary) failed after {retries + 1} attempts: {e}")
                return None
    return None # Should be unreachable if logic is correct

def get_openai_concise_summary(api_key, detailed_summary_text, timeout, retries):
    """Gets a concise summary from OpenAI's gpt-4o model.
    Includes timeout and retry logic.
    """
    if not api_key:
        st.error("OpenAI API Key is not provided.")
        return None
    if not detailed_summary_text:
        st.warning("Detailed summary (from Gemini) is not available for OpenAI processing.")
        return None
    
    client = openai.OpenAI(api_key=api_key)
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in summarizing clinical session transcripts."},
                    {"role": "user", "content": f"Condense the following detailed summary into a concise overview capturing all of its salient points. The output should be a single paragraph.\n\nDetailed Summary:\n{detailed_summary_text}"}
                ],
                temperature=0.7,
                max_tokens=500,
                timeout=timeout
            )
            concise_summary = response.choices[0].message.content.strip()
            return concise_summary
        except Exception as e:
            st.warning(f"OpenAI API (concise summary) attempt {attempt + 1} of {retries + 1} failed: {e}")
            if attempt < retries:
                time.sleep(1)  # Wait 1 second before retrying
            else:
                st.error(f"OpenAI API (concise summary) failed after {retries + 1} attempts: {e}")
                return None
    return None # Should be unreachable

def load_markdown_content(file_path):
    """Loads content from a markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Error: {file_path} not found.")
        return None
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")
        return None

def extract_json_from_markdown(markdown_content):
    """Extracts a JSON code block from markdown content."""
    match = re.search(r"```json\n(.*?)\n```", markdown_content, re.DOTALL)
    if match:
        json_string = match.group(1)
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON from schema example: {e}")
            return None
    else:
        st.warning("No JSON code block found in the schema example markdown. This is okay if the schema is defined programmatically.")
        return None

def get_gemini_ratings_and_justifications(api_key, transcript_text, ba_scale_content, schema_example_content, timeout, retries):
    """Gets ratings and justifications from Gemini using a predefined schema and context.
    Uses the model name 'gemini-2.5-pro-preview-05-06'.
    Includes timeout and retry logic.
    """
    if not api_key:
        st.error("Gemini API Key is not provided for ratings.")
        return None
    if not transcript_text:
        st.warning("Transcript text is not available for ratings.")
        return None
    if not ba_scale_content:
        st.error("BA Scale content (ba-scale.md) could not be loaded for ratings context.")
        return None
    if not schema_example_content: # This is the content of docs/schema.md, used for prompt context
        st.error("Schema example (docs/schema.md) could not be loaded for prompt context.")
        return None

    # Define the actual API response schema programmatically
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
                    "totalScore": {"type": "STRING"},  # Definition for nested totalScore
                    "meanScore": {"type": "STRING"}   # Definition for nested meanScore
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
                    "totalScore": {"type": "STRING"},  # Definition for nested totalScore
                    "meanScore": {"type": "STRING"}   # Definition for nested meanScore
                },
                "required": [
                    "rapportBuilding", "confidentiality", "activeListening", "openEndedQuestions",
                    "empathy", "collaborative", "validatesExperience", "encouraging",
                    "elicitsAffect", "summarizes", "totalScore", "meanScore"
                ]
            },
             # Adding totalScore and meanScore for interventionSpecificSkills and generalSkills here
            "interventionSpecificSkills_totalScore": {"type": "STRING"},
            "interventionSpecificSkills_meanScore": {"type": "STRING"},
            "generalSkills_totalScore": {"type": "STRING"},
            "generalSkills_meanScore": {"type": "STRING"},
            "totalMeanScore": {"type": "STRING"},
            "difficultyLevel": {"type": "STRING"},
            "audiotapeQuality": {"type": "STRING"},
            "providerOverallRating": {"type": "STRING"}
        },
        "required": [
            "providerName", "raterName", "trialId", "sessionDate", "ratingDate",
            "sessionNumber", "phaseNumber", "interventionSpecificSkills", "generalSkills",
            "interventionSpecificSkills_totalScore", "interventionSpecificSkills_meanScore", # ensure these are required
            "generalSkills_totalScore", "generalSkills_meanScore", # ensure these are required
            "totalMeanScore", "difficultyLevel", "audiotapeQuality", "providerOverallRating"
        ]
    }


    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        'gemini-2.5-pro-preview-05-06',
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=api_response_schema
        )
    )

    prompt = (
        "You are an expert rater evaluating a therapy session transcript based on the Behavior Activation (BA) Scale. "
        "Your task is to provide a rating (0-6, or N/A) and a concise justification for each skill listed in the BA scale. "
        "The justification should be specific to the provided transcript text and directly support the given rating. "
        "Refer to the BA Scale definitions provided below for guidance on each rating. "
        "Output your response strictly according to the provided JSON schema. "
        "Ensure all string fields in the JSON are properly escaped. "
        "Do not add any extra commentary or text outside of the JSON structure. "
        "The ratings should be based on the *entire transcript text*, not just a summary of it. "
        "For 'totalScore' and 'meanScore' under 'interventionSpecificSkills' and 'generalSkills' in the schema, calculate these based on your ratings for the skills in those respective sections. Ensure these are strings. If all items in a section are 'N/A', the scores can also be 'N/A'."
    )

    full_prompt = (
        f"{prompt}\n\n"
        f"TRANSCRIPT TEXT:\n{transcript_text}\n\n"
        f"BA SCALE DEFINITIONS:\n{ba_scale_content}\n\n"
        f"EXAMPLE JSON OUTPUT STRUCTURE (use this as a guide for the fields, but fill with actual ratings and justifications from the transcript. Note that the schema includes totalScore and meanScore fields directly under interventionSpecificSkills and generalSkills that you must calculate and populate, distinct from the skill-specific ratings and justifications.):\n{schema_example_content}"
    )
    
    for attempt in range(retries + 1):
        try:
            st.info(f"Sending request to Gemini for ratings (attempt {attempt + 1}/{retries + 1})...")
            response = model.generate_content(
                full_prompt,
                request_options={
                    "timeout": timeout
                }
            )
        
            if not response.parts:
                st.error("Gemini API returned no parts in the response for ratings.")
                if hasattr(response, 'prompt_feedback'):
                    st.error(f"Prompt Feedback: {response.prompt_feedback}")
                return None

            json_response_text = response.text # This should be the JSON string

            try:
                parsed_json = json.loads(json_response_text)
                return parsed_json
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse JSON response from Gemini: {e}")
                st.text_area("Problematic JSON Response from Gemini for Ratings:", value=json_response_text, height=300)
                if hasattr(response, 'prompt_feedback'):
                    st.error(f"Prompt Feedback: {response.prompt_feedback}")
                return None
            except Exception as e_gen: # Catch any other general errors during parsing or access
                 st.error(f"General error processing Gemini response for ratings: {e_gen}")
                 st.text_area("Problematic Full Response Object from Gemini:", value=str(response), height=300)
                 return None

        except Exception as e:
            st.warning(f"Gemini API (ratings) attempt {attempt + 1} of {retries + 1} failed: {e}")
            # Log more details if available from the exception
            if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
                st.warning(f"Prompt Feedback: {e.response.prompt_feedback}")
            elif hasattr(e, 'args') and len(e.args) > 0:
                 st.warning(f"Error details: {e.args[0]}")
            
            if attempt < retries:
                time.sleep(1)  # Wait 1 second before retrying
            else:
                st.error(f"Gemini API (ratings) failed after {retries + 1} attempts: {e}")
                return None
    return None # Should be unreachable

def parse_ba_scale_to_dict(ba_scale_content):
    """Parses the ba-scale.md content into a dictionary. 
    Maps skill_key to {item_name, description}.
    Relies on skill keys being explicitly defined in parentheses in the headings.
    """
    skills_dict = {}
    if not ba_scale_content:
        st.error("BA Scale content is empty, cannot parse.")
        return skills_dict

    # Regex to find skill blocks, now capturing the explicit skill_key from parentheses.
    # Example: ### 1. Uses the BA model (usesBAmodel)
    pattern = re.compile(
        r"### \d+\. (.*?) \((.*?)\)\n"  # Group 1: Item Name, Group 2: Skill Key
        r"\s*?"
        r"\*\*Description:\*\* (.*?)"
        r"(?=\n\s*\*\*Rating:\*\*)",
        re.DOTALL
    )

    matches = pattern.findall(ba_scale_content)
    for match in matches:
        item_name = match[0].strip()
        skill_key = match[1].strip() # Explicitly captured skill_key
        description_text = match[2].strip().replace('\n', ' ')
        
        if skill_key and item_name and description_text:
            skills_dict[skill_key] = {
                "item_name": item_name,
                "description": description_text
            }
        else:
            # This case should be less likely now with explicit keys
            st.warning(f"Could not fully parse a skill block. Title: '{item_name}', Captured Key: '{skill_key}'")

    if not skills_dict:
        st.warning("Could not parse any skills from ba-scale.md. Ensure skills have explicit (keysInParentheses) in headings.")
        st.text_area("BA Scale Content for Regex Debug:", ba_scale_content, height=200)
        st.text_area("Regex Used:", pattern.pattern, height=70)
    else:
        st.success(f"Successfully parsed {len(skills_dict)} skills from ba-scale.md.")

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

    # Define styles for appendix
    appendix_body_style = ParagraphStyle('AppendixBody', parent=body_style)
    appendix_body_style.spaceBefore = 2 # Tighter spacing for appendix
    appendix_body_style.spaceAfter = 2
    appendix_list_item_style = ParagraphStyle('AppendixListItem', parent=appendix_body_style, leftIndent=0.25*inch)

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
        # Check if detailed_summary_text is not None and not the placeholder string, and also not empty after strip
        if detailed_summary_text and detailed_summary_text.strip() and detailed_summary_text != "Detailed summary not available for appendix.":
            story.append(PageBreak())
            story.append(Paragraph("Appendix: Detailed Session Summary", title_style))
            story.append(Spacer(1, 0.1*inch))
            
            summary_lines = detailed_summary_text.split('\n')
            for line_text in summary_lines:
                current_content = line_text.strip()

                if not current_content:
                    leading_for_spacer = appendix_body_style.leading if hasattr(appendix_body_style, 'leading') and appendix_body_style.leading is not None else 12
                    story.append(Spacer(1, leading_for_spacer))
                    continue

                # Apply bold conversion first
                current_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', current_content)

                current_style_to_use = appendix_body_style
                
                # Check for unordered list items: * item or - item
                ul_match = re.match(r'^([\*\-])\s+(.*)', current_content)
                if ul_match:
                    content_after_marker = ul_match.group(2)
                    current_content = f"• {content_after_marker}"
                    current_style_to_use = appendix_list_item_style
                else:
                    # Check for ordered list items: 1. item
                    ol_match = re.match(r'^(\d+\.)\s+(.*)', current_content)
                    if ol_match:
                        number_marker = ol_match.group(1)
                        content_after_marker = ol_match.group(2)
                        current_content = f"{number_marker} {content_after_marker}" # Keep original numbering
                        current_style_to_use = appendix_list_item_style
                
                story.append(Paragraph(current_content, current_style_to_use))
    
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
            st.session_state.file_statuses = {} # Initialize file_statuses for UI
            num_files = len(uploaded_files)
            progress_bar = st.progress(0) # Initialize progress bar
            status_placeholder = st.empty() # Placeholder for detailed statuses

            ba_scale_doc_path = "docs/ba-scale.md"
            schema_doc_path = "docs/schema.md" # This is the EXAMPLE schema for the prompt
            
            ba_scale_content = load_markdown_content(ba_scale_doc_path)
            schema_example_content = load_markdown_content(schema_doc_path)

            # Parse BA scale definitions once
            ba_scale_dictionary = parse_ba_scale_to_dict(ba_scale_content)
            if not ba_scale_dictionary:
                st.error("Failed to parse ba-scale.md. Critical for table generation. Processing aborted for all files.")
                # return # or st.stop() if preferred to halt execution
            
            if not ba_scale_content or not schema_example_content: # ba_scale_content check is somewhat redundant now
                st.error("Failed to load critical document files (ba-scale.md or schema.md). Processing aborted.")
            else:
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    st.write(f"Processing {filename}...")

                    # Initialize status for the current file for UI
                    st.session_state.file_statuses[filename] = {
                        "detailed_summary_status": "Pending",
                        "ratings_status": "Pending",
                        "concise_summary_status": "Pending",
                        "pdf_report_status": "Pending"
                    }
                    update_progress_ui(progress_bar, status_placeholder, st.session_state.file_statuses, num_files)

                    current_file_result = {
                        "file": filename,
                        "detailed_summary": None,
                        "summary": None, # This will hold concise summary, or detailed if concise fails
                        "ratings": None,
                        "pdf_table_data": None,
                        "report_pdf_path": None
                    }

                    extracted_text = extract_text_from_pdf(uploaded_file)
                    if extracted_text:
                        st.session_state.file_statuses[filename]["detailed_summary_status"] = "In Progress"
                        update_progress_ui(progress_bar, status_placeholder, st.session_state.file_statuses, num_files)
                        detailed_summary = get_gemini_detailed_summary(
                            gemini_api_key, 
                            extracted_text, 
                            processing_timeout, 
                            retry_count
                        )
                        if detailed_summary:
                            current_file_result["detailed_summary"] = detailed_summary
                            current_file_result["summary"] = detailed_summary # Default to detailed
                            st.session_state.file_statuses[filename]["detailed_summary_status"] = "Complete"
                        else:
                            st.session_state.file_statuses[filename]["detailed_summary_status"] = "Error"
                            st.session_state.file_statuses[filename]["ratings_status"] = "Skipped (Summary Error)"
                            st.session_state.file_statuses[filename]["concise_summary_status"] = "Skipped (Summary Error)"
                            st.session_state.file_statuses[filename]["pdf_report_status"] = "Skipped (Summary Error)"
                            results.append(current_file_result) # Append even if error to show in results
                            update_progress_ui(progress_bar, status_placeholder, st.session_state.file_statuses, num_files)
                            continue 

                        # Step 9: Get ratings and justifications from Gemini
                        st.session_state.file_statuses[filename]["ratings_status"] = "In Progress"
                        update_progress_ui(progress_bar, status_placeholder, st.session_state.file_statuses, num_files)
                        ratings_justifications_json = get_gemini_ratings_and_justifications(
                            gemini_api_key,
                            extracted_text, 
                            ba_scale_content,
                            schema_example_content,
                            processing_timeout,
                            retry_count
                        )
                        if ratings_justifications_json:
                            current_file_result["ratings"] = ratings_justifications_json
                            st.session_state.file_statuses[filename]["ratings_status"] = "Complete"
                            # Display JSON for verification (optional, can be removed)
                            st.subheader(f"Ratings & Justifications (JSON) for {filename}:")
                            st.json(ratings_justifications_json) 
                        else:
                            st.session_state.file_statuses[filename]["ratings_status"] = "Error"
                            st.session_state.file_statuses[filename]["pdf_report_status"] = "Skipped (Ratings Error)"
                            results.append(current_file_result)
                            update_progress_ui(progress_bar, status_placeholder, st.session_state.file_statuses, num_files)
                            continue

                        # Step 10: Get Concise Summary from OpenAI (if API key provided)
                        if openai_api_key and current_file_result["detailed_summary"]:
                            st.session_state.file_statuses[filename]["concise_summary_status"] = "In Progress"
                            update_progress_ui(progress_bar, status_placeholder, st.session_state.file_statuses, num_files)
                            concise_summary = get_openai_concise_summary(
                                openai_api_key, 
                                current_file_result["detailed_summary"], # Use from current_file_result
                                processing_timeout, 
                                retry_count
                            )
                            if concise_summary:
                                current_file_result["summary"] = concise_summary # Update summary to concise
                                st.session_state.file_statuses[filename]["concise_summary_status"] = "Complete"
                                # Display concise summary (optional)
                                st.subheader(f"Concise Summary for {filename}:")
                                st.markdown(concise_summary)
                            else:
                                st.session_state.file_statuses[filename]["concise_summary_status"] = "Error (Using Detailed)"
                                # current_file_result["summary"] already holds detailed_summary
                        else:
                            if not openai_api_key:
                                st.session_state.file_statuses[filename]["concise_summary_status"] = "Skipped (No API Key)"
                            else: # No detailed summary to work with
                                st.session_state.file_statuses[filename]["concise_summary_status"] = "Skipped (No Detailed Summary)"

                        update_progress_ui(progress_bar, status_placeholder, st.session_state.file_statuses, num_files)

                        # Step 11: Assemble table data (ensure ratings are available)
                        if current_file_result["ratings"] and ba_scale_dictionary:
                            pdf_table_data = prepare_data_for_pdf_tables(current_file_result["ratings"], ba_scale_dictionary)
                            current_file_result["pdf_table_data"] = pdf_table_data
                            # Display table data (optional)
                            st.subheader(f"Prepared Data for PDF Tables (File: {filename})")
                            st.json(pdf_table_data) 
                        else:
                            if not current_file_result["ratings"]:
                                st.warning(f"Skipping PDF table data for {filename} as ratings are missing.")
                            # ba_scale_dictionary failure is handled earlier
                            st.session_state.file_statuses[filename]["pdf_report_status"] = "Skipped (Table Data Error)"
                            results.append(current_file_result) # Append partial result
                            update_progress_ui(progress_bar, status_placeholder, st.session_state.file_statuses, num_files)
                            continue 

                        # Step 12: Generate PDF Report (ensure table data is available)
                        if current_file_result["pdf_table_data"]:
                            st.session_state.file_statuses[filename]["pdf_report_status"] = "In Progress"
                            update_progress_ui(progress_bar, status_placeholder, st.session_state.file_statuses, num_files)
                            original_fname_base, _ = os.path.splitext(filename)
                            report_path = generate_pdf_report(current_file_result, original_fname_base, include_detailed_summary_appendix)
                            if report_path:
                                current_file_result["report_pdf_path"] = report_path
                                st.session_state.file_statuses[filename]["pdf_report_status"] = "Complete"
                                with open(report_path, "rb") as pdf_file_to_download:
                                    st.download_button(
                                        label=f"Download Report for {filename}",
                                        data=pdf_file_to_download,
                                        file_name=os.path.basename(report_path),
                                        mime="application/pdf",
                                        key=f"download_{filename}" # Unique key for download button
                                    )
                            else:
                                st.session_state.file_statuses[filename]["pdf_report_status"] = "Error"
                        else:
                             st.warning(f"Skipping PDF report generation for {filename} as table data is missing.")
                             st.session_state.file_statuses[filename]["pdf_report_status"] = "Skipped (No Table Data)"

                    else: # Extracted text failed
                        st.warning(f"Could not extract text from {filename}. Skipping further processing.")
                        st.session_state.file_statuses[filename]["detailed_summary_status"] = "Skipped (Extraction Error)"
                        st.session_state.file_statuses[filename]["ratings_status"] = "Skipped"
                        st.session_state.file_statuses[filename]["concise_summary_status"] = "Skipped"
                        st.session_state.file_statuses[filename]["pdf_report_status"] = "Skipped"
                   
                    results.append(current_file_result) # Append result for the file
                    update_progress_ui(progress_bar, status_placeholder, st.session_state.file_statuses, num_files)
               
                # Final update to progress UI after loop
                update_progress_ui(progress_bar, status_placeholder, st.session_state.file_statuses, num_files)

                if any(r.get("report_pdf_path") for r in results): 
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