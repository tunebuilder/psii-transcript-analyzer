# PSii Transcription Review & Rating App

## Overview

This application helps clinicians and researchers streamline the review of therapy session transcripts. It takes PDF transcripts of clinical sessions, uses AI to analyze the content, and generates a structured report based on the Behavioral Activation (BA) quality scale. The goal is to save time and provide consistent, AI-assisted feedback and ratings.

## Key Features

*   **PDF Upload**: Easily upload one or more PDF transcripts of clinical sessions.
*   **Automated Text Extraction**: Extracts the text content from your PDF files.
*   **AI-Powered Summaries**:
    *   Generates a detailed summary of each session, highlighting clinician performance and trainer feedback (using Google's Gemini model).
    *   Provides a concise, single-paragraph overview of the session (using OpenAI's GPT-4o model).
*   **Structured BA Scale Ratings**:
    *   Utilizes the Gemini model to rate the session against the items in the Behavioral Activation (BA) quality scale.
    *   Provides justifications for each rating, referencing the session content.
    *   Fills in metadata related to the session and provider.
*   **Downloadable PDF Reports**:
    *   For each processed transcript, a comprehensive PDF report is generated.
    *   Reports include the concise summary, detailed BA scale ratings (with descriptions, ratings, and justifications in table format), and overall assessment scores.
    *   Optionally, users can choose to include the full detailed AI summary as an appendix to the PDF report.
*   **User-Friendly Interface**:
    *   Built with Streamlit for an interactive web application experience.
    *   Sidebar for entering necessary API keys (Gemini, OpenAI) and adjusting processing settings.

## How to Use

1.  **Installation**:
    *   Ensure you have Python installed.
    *   Clone this repository (if applicable) or ensure you have all project files.
    *   Install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```

2.  **Run the Application**:
    *   Open your terminal or command prompt.
    *   Navigate to the directory containing `app.py`.
    *   Run the Streamlit app:
        ```bash
        streamlit run app.py
        ```
    *   The application will open in your web browser.

3.  **Enter API Keys**:
    *   In the sidebar, enter your Google Gemini API Key and your OpenAI API Key. These are required for the AI processing features.

4.  **Upload PDFs**:
    *   Use the file uploader in the main area to select one or more PDF transcript files.

5.  **Start Processing**:
    *   Click the "Start Processing" button.
    *   The app will process each file sequentially, displaying summaries and ratings in the UI.

6.  **Download Reports**:
    *   Once processing for a file is complete, a download button for its PDF report will appear.
    *   The generated PDF reports will also be saved in an `outputs/` directory in the project folder.

## Core Technologies

*   **Python**: The core programming language.
*   **Streamlit**: For building the interactive web application UI.
*   **Google Generative AI (Gemini)**: For detailed summaries and BA scale ratings.
*   **OpenAI API (GPT-4o)**: For concise summaries.
*   **PyPDF2**: For extracting text from PDF files.
*   **ReportLab**: For generating the PDF reports.
*   **Pandas**: For data manipulation (used internally).

---
*This application is designed to assist with the review process and should be used in conjunction with professional clinical judgment.* 