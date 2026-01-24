ITERATIVE_PROMPTING_PROMPT_OPENING_CHUNK_SELECTION = """\
You are given:
1. A factual statement.
2. A list of numbered sentences extracted from source documents.

Identify all sentence numbers from the list that directly and explicitly relate to the factual statement.
A sentence relates to the statement if it clearly affirms it, contradicts it, or provides factual evidence about it, without requiring inference.

Return ONLY the list of sentence numbers in square brackets, for example: [2, 4, 5].
If no sentences are relevant, return [].
Do NOT include any explanations, additional text, or characters outside this exact format.
"""

ITERATIVE_PROMPTING_PROMPT_OPENING_WITH_USED_CHUNK_SELECTION = """\
You are given:
1. A factual statement.
2. A list of numbered sentences extracted from source documents.
3. A list of sentences labeled "used" (given as raw text, not sentence numbers) that have already been selected as relevant evidence.

Identify additional sentence numbers from the numbered list that, when combined with the "used" sentences, directly and explicitly relate to the factual statement.
Include only sentences that add new, relevant, and direct factual evidence about the statement.
Do NOT re-select any sentences whose text appears in the "used" sentences list.

Return ONLY the list of newly selected sentence numbers in square brackets, for example: [2, 4, 5].
If no additional sentences apply, return [].
Do NOT include explanations, text, or any characters outside this exact format.
"""

ITERATIVE_PROMPTING_PROMPT_OPENING_ONE_SELECTION = """\
You are given:
1. A factual statement.
2. A list of numbered sentences extracted from source documents.

Identify exactly one sentence number from the list that directly and explicitly relates to the factual statement.
A sentence relates to the statement if it clearly provides evidence about it, without requiring inference. 
Select the single most relevant sentence from the list, choosing the one that provides the clearest evidence.

### OUTPUT FORMAT RULES ###
- Output MUST consist of ONLY one sentence number in square brackets.
- Example: [2]
- Do NOT output explanations, reasoning, or any extra text.
- Your ENTIRE response must exactly match the required format.
"""

ITERATIVE_PROMPTING_PROMPT_OPENING_WITH_USED_ONE_SELECTION = """\
You are given:
1. A factual statement.
2. A list of numbered sentences extracted from source documents.
3. A list of sentences labeled "used" (given as raw text, not sentence numbers) that have already been selected as relevant evidence.

Identify exactly one additional sentence number from the numbered list that, when combined with the "used" sentences, directly and explicitly relates to the factual statement.
Do NOT re-select any sentences whose text appears in the "used" sentences list.
Select the single most relevant new sentence from the list, choosing the one that provides the clearest evidence.

### OUTPUT FORMAT RULES ###
- Output MUST consist of ONLY one newly selected sentence number in square brackets.
- Example: [2]
- Do NOT output explanations, reasoning, or any extra text.
- Your ENTIRE response must exactly match the required format.
"""

LLM_RANKING_PROMPT = """\
You are given:
1. A factual statement.
2. A list of exactly {NUMBER_OF_SENTENCES} numbered sentences extracted from source documents. Sentence IDs start at 1.

Rank all {NUMBER_OF_SENTENCES} sentences by how directly and clearly they relate to the factual statement. 
A sentence relates to the statement if it clearly provides evidence about it, without requiring inference. 
Rank from the most directly informative sentence to the least, and include all sentences.

### OUTPUT FORMAT RULES ###
- Output MUST be a JSON object with sentence IDs as keys and sentence texts as values.
- The order of the key-value pairs in the JSON represents the ranking: the first pair is the most relevant sentence, the second is next, and so on.
- Example: {{"1": "Sentence text 1", "5": "Sentence text 5", "3": "Sentence text 3"}} means sentence 1 is ranked first, sentence 5 second, and sentence 3 third.
- Do NOT output explanations, reasoning, or any extra text.
- Your ENTIRE response must exactly match the required format.

Statement: {claim}
Sentences:
{sentences}
"""

LLM_EVIDENCE_SELECTION_PROMPT = """\
You are given:
1. A factual statement.
2. A list of exactly {NUMBER_OF_SENTENCES} numbered sentences extracted from source documents. Sentence IDs start at 1.

Your task:
Select ONLY the sentences that directly help verify the factual statement (either supporting or refuting it).
A sentence helps verify the statement if it clearly provides evidence about it **without any inference**.

The selected set of sentences MUST:
- Be sufficient to fully cover the factual statement.
- Include only sentences that directly support or directly refute the statement.
- Exclude any irrelevant or weakly related sentences.

### OUTPUT FORMAT RULES ###
- Output MUST be a Python-style array of sentence IDs (integers) only.
- Include ONLY the sentences you select as evidence.
- Do NOT rank them; order does not matter.
- Do NOT output explanations, reasoning, or extra text.
- Your ENTIRE response must exactly match the required format.

### EXAMPLE OUTPUT ###
[2, 14]

Statement: {claim}
Sentences:
{sentences}
"""

REASONRANK_PROMPT = """You are RankLLM, an intelligent assistant that can rank sentences based on their relevance to a factual statement.
Given a factual statement and a sentence list, you first thinks about the reasoning process in the mind and then
provides the answer (i.e., the reranked sentence list). The reasoning process and answer are enclosed
within <think> and <answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
I will provide you with {num} sentences, each indicated by a numerical identifier [].
Rank the sentences based on their relevance to the factual statement: {claim_sentence}.

{sentences_formatted}

Factual Statement: {claim_sentence}. Rank the {num} sentences above based on their relevance to the factual statement.

All the sentences should be included and listed using identifiers, in descending order of relevance.
The format of the answer should be [] > [], e.g., [2] > [1].
"""