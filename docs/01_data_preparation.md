# Gillam Corpus Data Preparation Summary

## Overview
Successfully extracted and prepared child narrative data from the Gillam corpus for word embedding analysis. The corpus contains transcribed narratives from 668 children, including both typically developing (TD) children and those with specific language impairment (SLI).

## Final Dataset Statistics

### Corpus Composition
- **Total children**: 668
  - **TD children**: 497
  - **SLI children**: 171

### Age Distribution
- Ages range from 5 to 11 years
- Most data concentrated in ages 6-8 years
- Both TD and SLI groups have good representation across age ranges

### Data Volume
- **Average utterances per child**:
  - TD: 39.9 utterances
  - SLI: 34.3 utterances
- **Average words per child**:
  - TD: 364.6 words
  - SLI: 266.1 words
- **Mean Length of Utterance (MLU)**:
  - TD: 8.97 words per utterance (SD=1.47)
  - SLI: 7.60 words per utterance (SD=1.45)

## Data Processing Pipeline

### 1. Extraction (`extract_gillam_data.py`)
- Parsed 668 .cha files in CHAT format
- Extracted child utterances (marked with `*CHI:`)
- Removed researcher speech
- Extracted metadata (age, gender, development type, ethnicity)

### 2. Cleaning
- Removed CHAT-specific annotations:
  - Ampersand codes (e.g., `&-um`, `&~w`)
  - Bracketed annotations (`[/]`, `[//]`, `[*]`)
  - Angle bracket content (`<word>`)
  - Repetition markers
- Normalized whitespace and punctuation
- Preserved semantic content while removing transcription artifacts

### 3. Validation (`validate_and_improve_cleaning.py`)
- Verified all files extracted successfully
- Applied enhanced cleaning to remove remaining artifacts
- Created embedding-ready format with standardized filenames

### 4. MLU Calculation (`add_mlu_to_metadata.py`)
- Calculated Mean Length of Utterance for each child
- Added MLU column to all metadata files
- Identified MLU-matched pairs for controlled comparisons

## Output Structure

### Directory Layout
```
extracted_data/
├── child_texts/              # Raw extracted texts
├── child_texts_enhanced/     # Enhanced cleaned texts
├── embedding_ready/          # Final texts ready for embedding
│   ├── *.txt                 # Individual child texts
│   └── metadata.csv          # Complete metadata
├── metadata.csv              # Original extraction metadata
└── combined_*.txt            # Optional combined group files
```

### File Naming Convention
Embedding-ready files use standardized names:
`{dev_type}_{age}y{months}m_{gender}_{original_id}.txt`

Example: `td_7y3m_f_46905ca.txt`
- `td` = typically developing
- `7y3m` = 7 years, 3 months old
- `f` = female
- `46905ca` = original child ID

## Metadata Fields
The `metadata.csv` file contains:
- `filename`: Standardized filename
- `original_id`: Original child identifier
- `development_type`: TD or SLI
- `age_years`: Age in years
- `age_months`: Additional months
- `gender`: male/female
- `ethnicity`: When available
- `num_utterances`: Count of utterances
- `total_words`: Total word count
- `mlu`: Mean Length of Utterance (words per utterance)
- `text_path`: Full path to text file

## Ready for Embedding Training

The data is now prepared for your word embedding analysis:

1. **Individual child texts** are in `extracted_data/embedding_ready/`
2. **Metadata** for analysis is in `extracted_data/embedding_ready/metadata.csv`
3. Each child has a clean text file containing only their speech
4. Files are small (avg ~300-400 words) but suitable for specialized embedding techniques

## MLU Analysis Highlights

### MLU Distribution
- **Significant MLU overlap** between TD and SLI groups (3.90 - 11.38 range)
- **170 out of 171 SLI children** have potential MLU-matched TD peers (±0.5)
- This enables controlled comparisons where network measures can be tested independent of MLU

### MLU by Age (mean values)
| Age | TD MLU | SLI MLU | Difference |
|-----|--------|---------|------------|
| 5   | 7.79   | 5.66    | 2.13       |
| 6   | 8.09   | 6.84    | 1.25       |
| 7   | 8.95   | 7.55    | 1.40       |
| 8   | 9.16   | 7.56    | 1.60       |
| 9   | 9.54   | 8.39    | 1.15       |
| 10  | 10.01  | 8.84    | 1.17       |
| 11  | 9.92   | 8.57    | 1.35       |

## Next Steps for Your Research

For your semantic network analysis:
1. Train word embeddings for each individual child using the texts in `embedding_ready/`
2. Use the metadata to group children by development type (TD/SLI) and age
3. **Create MLU-matched pairs** using the MLU column for controlled comparisons
4. Construct semantic networks from the embeddings
5. Analyze network properties (hub prominence, clustering coefficients)
6. Compare network structures between TD and SLI groups
7. **Test whether network measures can distinguish TD from SLI children with similar MLU**

The clean, individual text files with MLU data allow you to:
- Train child-specific embeddings
- Control for age, gender, and MLU effects
- Compare semantic organization between TD and SLI children
- Test your hypothesis about small-world network disruption in SLI
- **Demonstrate that network measures may outperform traditional MLU-based diagnostics**

## Notes
- Some children have limited data (<50 words) - consider minimum thresholds
- Age-matched comparisons are possible with good sample sizes
- Consider using techniques suitable for small corpora (e.g., contextualized embeddings, few-shot learning)