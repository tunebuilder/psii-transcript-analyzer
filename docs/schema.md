Create a structured output using the Behavioral Activation (BA) quality scale, ensuring that item ratings are nested under two sections: intervention-specific skills and general skills.

Include the following fields for each section:
- Provider's Name
- Rater's Name
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
  "providerName": "Provider's Name",
  "raterName": "Rater's Name",
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
- Use 'N/A' where an item is not applicable.
- The total scores should reflect the sum of all rated items within their respective skills section.
- Mean scores are calculated by dividing the total score by the number of applicable items evaluated. 