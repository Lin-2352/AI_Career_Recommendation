# Datasets

## Active Training Sources

| File | Use | Notes |
| --- | --- | --- |
| `data/raw/career_guidance.csv` | Active | University career guidance records mapped into education, skill, interest, and target-career fields. |
| `data/raw/career_recommendation.csv` | Active | Candidate skills and interests with recommended career labels. |
| `data/raw/student_scores_sanitized.csv` | Active | Sanitized academic-score data. Names, emails, IDs, and gender are removed before training. |

## Excluded Local Files

| File | Reason |
| --- | --- |
| `data/raw/student-scores-6k.csv` | Contains direct personal identifiers and is replaced by `student_scores_sanitized.csv`. |
| `data/raw/Job Datsset.csv` | Contains job-match rows but no career target label for supervised career classification. |
| `data/raw/computer_science_student_career_datasetMar62024.csv` | Validation showed that adding it sharply reduced recommendation quality, so it is treated as experimental local data. |

## Current Training Result

The active training set contains career guidance, candidate recommendation, and sanitized student-score data. The excluded datasets remain local-only and are not pushed to GitHub.
