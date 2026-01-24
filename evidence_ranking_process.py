import ast
import copy
import re
from collections import Counter

import numpy as np
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams

import constants.constants as consts
from constants.prompt_constants import *
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SENTENCE_TRANSFORMER_MODEL = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)

tokenizer = None
model = None

if any(evidence_ranking_method in EVIDENCE_RANKINGS_METHODS for evidence_ranking_method in [NLI_BASED_RANKING, ITERATIVE_NLI_BASED_RANKING]):
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)


def perform_citation_selection(claim_to_candidate_evidence_mapping_instances, evidence_ranking_method, should_create_gold_ranking_json):
    instances_with_ranked_evidence = {}
    instances_with_ranked_gold_snippets = {}

    if evidence_ranking_method == ONE_SHOT_REASON_RANK_MODEL:
        llm = LLM(model=RERANKER_MODEL_NAME, dtype="bfloat16")
    else:
        llm = None

    for instance_id, claim_to_candidate_evidence_mapping_dict in claim_to_candidate_evidence_mapping_instances.items():
        if claim_to_candidate_evidence_mapping_dict:
            perform_evidence_ranking_for_instance(instance_id, claim_to_candidate_evidence_mapping_dict,
                                                  evidence_ranking_method, instances_with_ranked_evidence,
                                                  instances_with_ranked_gold_snippets,
                                                  should_create_gold_ranking_json, llm)

    return instances_with_ranked_evidence, instances_with_ranked_gold_snippets


def perform_evidence_ranking_for_instance(instance_id, claim_to_candidate_evidence_mapping_dict,
                                          evidence_ranking_method, instances_with_ranked_evidence,
                                          instances_with_ranked_gold_snippets, should_create_gold_ranking_json, llm):
    ranked_evidence_dict = get_top_supporting_citations_for_instance(claim_to_candidate_evidence_mapping_dict, evidence_ranking_method, llm)

    update_instances_json_with_ranked_evidence(instances_with_ranked_evidence, ranked_evidence_dict, instance_id)

    if should_create_gold_ranking_json and SHOULD_USE_GOLD_SNIPPETS:
        ranked_gold_evidence_dict = get_top_ranked_gold_snippets_for_instance(claim_to_candidate_evidence_mapping_dict)
        update_instances_json_with_ranked_evidence(instances_with_ranked_gold_snippets, ranked_gold_evidence_dict, instance_id)


def get_top_supporting_citations_for_instance(claim_to_candidate_evidence_mapping_dict, evidence_ranking_method, llm):
    ranked_claim_to_evidence_dict = {}

    for claim_sentence, candidate_evidence_sentences in claim_to_candidate_evidence_mapping_dict.items():
        candidate_evidence_list = candidate_evidence_sentences[CANDIDATE_SNIPPETS]

        if candidate_evidence_list:
            ranked_evidence_dict = get_ranked_evidence_for_claim(claim_sentence, candidate_evidence_list, evidence_ranking_method, llm)
        else:
            ranked_evidence_dict = {}

        ranked_evidence_dict[GOLD] = candidate_evidence_sentences[GOLD]
        ranked_evidence_dict[LABEL] = candidate_evidence_sentences[LABEL]

        ranked_claim_to_evidence_dict[claim_sentence] = ranked_evidence_dict

    return ranked_claim_to_evidence_dict


def get_ranked_evidence_for_claim(claim_sentence, candidate_evidence_list, evidence_ranking_method, llm):
    ranked_evidence_dict = {}

    match evidence_ranking_method:
        case consts.SIMILARITY_METHOD:
            ranked_evidence_dict = perform_similarity_for_evidence_ranking(claim_sentence, candidate_evidence_list)

        case consts.ITERATIVE_SIMILARITY_METHOD:
            ranked_evidence_dict = perform_iterative_similarity_for_evidence_ranking(claim_sentence, candidate_evidence_list)

        case consts.ONE_SHOT_PROMPTING:
            ranked_evidence_dict = perform_one_shot_prompting_for_evidence_ranking(claim_sentence,
                                                                                   candidate_evidence_list)

        case consts.ITERATIVE_PROMPTING_SENTENCE:
            ranked_evidence_dict = perform_iterative_prompting_for_evidence_ranking(claim_sentence,
                                                                                    candidate_evidence_list, False)

        case consts.NLI_BASED_RANKING:
            ranked_evidence_dict = perform_nli_ranking_for_evidence_ranking(claim_sentence, candidate_evidence_list)

        case consts.ONE_SHOT_REASON_RANK_MODEL:
            ranked_evidence_dict = perform_reasonrank_one_shot_with_window(claim_sentence, candidate_evidence_list, llm)

        case consts.LAQUER_SELECTION:
            ranked_evidence_dict = perform_laquer_selection(claim_sentence, candidate_evidence_list)

    return ranked_evidence_dict


def perform_iterative_similarity_for_evidence_ranking(claim_sentence, candidate_evidence_list):
    ranked_evidence_dict = {}

    claim_emb = get_embedding_for_sentence(claim_sentence)
    evidence_embs = get_embeddings_for_sentences(candidate_evidence_list)

    selected_indices = set()
    selected_evidence_list = []
    selected_embs = []

    for i in range(1, len(candidate_evidence_list) + 1):
        best_idx, best_score = None, -1

        for idx, emb in enumerate(evidence_embs):
            if idx in selected_indices:
                continue

            new_embs = selected_embs + [emb]
            avg_emb = sum(new_embs) / len(new_embs)

            sim_score = util.cos_sim(claim_emb, avg_emb).item()

            if sim_score > best_score:
                best_idx, best_score = idx, sim_score

        # update chosen set
        selected_indices.add(best_idx)
        selected_embs.append(evidence_embs[best_idx])
        selected_evidence_list.append(candidate_evidence_list[best_idx])

        ranked_evidence_dict[i] = {
            SELECTED_SNIPPETS: selected_evidence_list.copy(),
        }

    return ranked_evidence_dict


def perform_similarity_for_evidence_ranking(claim_sentence, candidate_evidence_list):
    ranked_evidence_dict = {}

    claim_emb = get_embedding_for_sentence(claim_sentence)
    evidence_embs = get_embeddings_for_sentences(candidate_evidence_list)

    scores = [(i, util.cos_sim(claim_emb, evidence_emb).item()) for i, evidence_emb in enumerate(evidence_embs)]
    scores.sort(key=lambda x: x[1], reverse=True)

    cumulative_snippets = []

    for rank, (idx, score) in enumerate(scores, start=1):
        cumulative_snippets.append(candidate_evidence_list[idx])

        ranked_evidence_dict[rank] = {
            SELECTED_SNIPPETS: cumulative_snippets.copy(),
        }

    return ranked_evidence_dict


def get_embedding_for_sentence(sentence):
    prompt = "Represent this sentence for retrieval: " + sentence
    sentence_emb = SENTENCE_TRANSFORMER_MODEL.encode(prompt, convert_to_tensor=True)

    return sentence_emb


def get_embeddings_for_sentences(sentences):
    prompts = ["Represent this sentence for retrieval: " + s for s in sentences]
    sentences_embs = SENTENCE_TRANSFORMER_MODEL.encode(prompts, convert_to_tensor=True)

    return sentences_embs


def parse_llm_evidence_ranking_response(llm_raw_response):
    evidence_numbers_list = []

    try:
        result = ast.literal_eval(llm_raw_response)
        if isinstance(result, list) and all(isinstance(x, int) for x in result):
            evidence_numbers_list = result
    except (ValueError, SyntaxError):
        pass

    if not evidence_numbers_list:
        matches = re.findall(r"\[(\d+)\]", llm_raw_response)
        if matches:
            # Keep consistent with your current design: return as list of ints
            evidence_numbers_list = [int(matches[-1])]

    return evidence_numbers_list


def perform_iterative_prompting_for_evidence_ranking(claim_sentence, candidate_evidence_list, is_chunk_selection):
    ranked_evidence_dict = {}
    used_evidence = []
    should_query = True
    attempt_num = 1

    while should_query:
        unused_evidence = list((Counter(candidate_evidence_list) - Counter(used_evidence)).elements())

        if not len(unused_evidence):
            break

        prompt = create_prompt_for_evidence_ranking_iterative_prompting(claim_sentence, unused_evidence,
                                                                        used_evidence, is_chunk_selection)
        messages_list = create_messages_for_query(prompt)

        if PLATFORM_TYPE == OPEN_AI_PLATFORM_TYPE:
            llm_raw_response, is_query_failed = query_openai_model(messages_list)
        elif PLATFORM_TYPE == TOGETHER_AI_PLATFORM_TYPE:
            llm_raw_response, is_query_failed = query_togetherai_model(messages_list)
        elif PLATFORM_TYPE == OPENROUTER_PLATFORM_TYPE:
            llm_raw_response, is_query_failed = query_openrouter_model(messages_list)

        if is_query_failed:
            ranked_evidence_dict[attempt_num] = {SELECTED_SNIPPETS: []}
            break
        else:
            evidence_numbers_list = parse_llm_evidence_ranking_response(llm_raw_response)

            if evidence_numbers_list:
                new_ranked_evidence_list = [unused_evidence[i - 1] for i in evidence_numbers_list if
                                            1 <= i <= len(unused_evidence)]

                if new_ranked_evidence_list:
                    ranked_evidence_list = used_evidence + new_ranked_evidence_list
                    ranked_evidence_dict[attempt_num] = {SELECTED_SNIPPETS: ranked_evidence_list.copy()}
                    used_evidence.extend(new_ranked_evidence_list)
                    attempt_num += 1
                else:
                    evidence_numbers_list = []
                    ranked_evidence_list = []
            else:
                ranked_evidence_list = []

        if evidence_numbers_list == []:
            if len(ranked_evidence_list) != len(candidate_evidence_list):
                ranked_evidence_list = list(used_evidence)
                for next_selected_sentence in unused_evidence:
                    ranked_evidence_list.append(next_selected_sentence)
                    ranked_evidence_dict[attempt_num] = {SELECTED_SNIPPETS: ranked_evidence_list.copy()}
                    attempt_num += 1

            should_query = False

    return ranked_evidence_dict


def create_prompt_for_evidence_ranking_iterative_prompting(claim_sentence, unused_evidence, used_evidence,
                                                           is_chunk_selection):
    prompt_statement = f"Statement: {claim_sentence}"
    numbered_evidence_sentences = 'Sentences:\n' + '\n'.join(f"{i + 1}. {s}" for i, s in enumerate(unused_evidence))

    if used_evidence:
        if is_chunk_selection:
            answer_format = "Answer Format (example only):\n[2, 4, 5]"
            prompt_opening = ITERATIVE_PROMPTING_PROMPT_OPENING_WITH_USED_CHUNK_SELECTION
        else:
            answer_format = "Answer Format (example only):\n[2]"
            prompt_opening = ITERATIVE_PROMPTING_PROMPT_OPENING_WITH_USED_ONE_SELECTION

        used_evidence_string = 'Used sentences:\n' + '\n'.join(f"- {s}" for s in used_evidence)
        prompt = prompt_opening + "\n" + prompt_statement + "\n\n" + used_evidence_string + "\n\n" + numbered_evidence_sentences + "\n\n" + answer_format
    else:
        if is_chunk_selection:
            answer_format = "Answer Format (example only):\n[2, 4, 5]"
            prompt_opening = ITERATIVE_PROMPTING_PROMPT_OPENING_CHUNK_SELECTION
        else:
            answer_format = "Answer Format (example only):\n[2]"
            prompt_opening = ITERATIVE_PROMPTING_PROMPT_OPENING_ONE_SELECTION

        prompt = prompt_opening + "\n" + prompt_statement + "\n\n" + numbered_evidence_sentences + "\n\n" + answer_format

    return prompt


def perform_one_shot_prompting_for_evidence_ranking(claim_sentence, candidate_evidence_list):
    ranked_evidence_dict = {}
    should_query = True
    attempt_num = 1

    feedback_message = None
    assistant_message = None

    while should_query:
        prompt = create_prompt_for_evidence_ranking_one_shot_prompting(claim_sentence, candidate_evidence_list)

        messages_list = create_messages_for_query(prompt, feedback_message, assistant_message)

        if PLATFORM_TYPE == OPEN_AI_PLATFORM_TYPE:
            llm_raw_response, is_query_failed = query_openai_model(messages_list)
        elif PLATFORM_TYPE == TOGETHER_AI_PLATFORM_TYPE:
            llm_raw_response, is_query_failed = query_togetherai_model(messages_list)
        elif PLATFORM_TYPE == OPENROUTER_PLATFORM_TYPE:
            llm_raw_response, is_query_failed = query_openrouter_model(messages_list)

        evidence_list = []

        if is_query_failed:
            if attempt_num <= MAX_ATTEMPTS_FOR_API_QUERY:
                attempt_num += 1
                continue
        else:
            evidence_dict = safe_str_to_json(llm_raw_response)

            if not evidence_dict:
                if attempt_num <= MAX_ATTEMPTS_FOR_API_QUERY:
                    feedback_message = "The previous output was not valid JSON. Please output EXACTLY a JSON object as instructed, without any extra text, explanations, or formatting errors."
                    attempt_num += 1
                    continue

            keys_set = {int(k) for k in evidence_dict.keys()}

            if keys_set == set(range(1, len(candidate_evidence_list)+1)):
                evidence_list = []
                for idx, (key, _) in enumerate(evidence_dict.items(), start=1):
                    sentence_evidence = candidate_evidence_list[int(key)-1]
                    evidence_list.append(sentence_evidence)
                    ranked_evidence_dict[idx] = {SELECTED_SNIPPETS: evidence_list.copy()}

                should_query = False
            else:
                if attempt_num <= MAX_ATTEMPTS_FOR_API_QUERY:
                    feedback_message = f"The previous output did not include all {len(candidate_evidence_list)} sentences. Make sure to include every sentence ID from 1 to {len(candidate_evidence_list)} in the JSON, even if some are less relevant."
                    attempt_num += 1
                    continue

        if not evidence_list:
            for idx, sentence in enumerate(candidate_evidence_list, start=1):
                evidence_list.append(sentence)
                ranked_evidence_dict[idx] = {SELECTED_SNIPPETS: evidence_list.copy()}
            should_query = False

    return ranked_evidence_dict


def create_prompt_for_evidence_ranking_one_shot_prompting(claim_sentence, candidate_evidence_list):
    numbered_candidate_evidence_sentences = '\n'.join(f"{i + 1}. {s}" for i, s in enumerate(candidate_evidence_list))
    number_of_sentences = len(candidate_evidence_list)

    prompt = LLM_RANKING_PROMPT.format(
        NUMBER_OF_SENTENCES=number_of_sentences,
        claim=claim_sentence,
        sentences=numbered_candidate_evidence_sentences
    )

    return prompt


def perform_nli_ranking_for_evidence_ranking(claim_sentence, candidate_evidence_list, top_k=2, max_tokens=512):
    scores = compute_sentence_scores(claim_sentence, candidate_evidence_list, max_tokens=max_tokens)
    strongest_label = nli_topk_rerank(claim_sentence, scores, top_k=top_k, max_tokens=max_tokens)

    ranked_sentences = sorted(scores, key=lambda x: x[strongest_label], reverse=True)
    ranked_sentences_list = [sentence_dict["sentence"] for sentence_dict in ranked_sentences]
    ranked_evidence_dict = create_accumulative_mapping_for_instance(ranked_sentences_list)

    return ranked_evidence_dict


def compute_sentence_scores(claim_sentence, candidate_evidence_list, max_tokens=512):
    scores = []

    for sent in candidate_evidence_list:
        encoded = tokenizer.encode_plus(
            sent, claim_sentence,
            return_tensors='pt', truncation=True, max_length=max_tokens
        )
        for key in encoded:
            encoded[key] = encoded[key].to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        scores.append({
            'sentence': sent,
            'entailment': probs[ENTAILMENT_INDEX],
            'contradiction': probs[CONTRADICTION_INDEX]
        })

    return scores


def nli_topk_rerank(claim_sentence, scores, top_k=2, max_tokens=512):
    top_entail = sorted(scores, key=lambda x: x['entailment'], reverse=True)[:top_k]
    top_contra = sorted(scores, key=lambda x: x['contradiction'], reverse=True)[:top_k]

    rerank_inputs = [top_entail + top_contra, top_contra + top_entail]

    final_scores = []

    for concat_order in rerank_inputs:
        multi_premise = " ".join([s['sentence'] for s in concat_order])
        encoded = tokenizer.encode_plus(
            multi_premise, claim_sentence,
            return_tensors='pt', truncation=True, max_length=max_tokens
        )

        for key in encoded:
            encoded[key] = encoded[key].to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        final_scores.append(probs)

    avg_probs = np.mean(final_scores, axis=0)

    strongest_label = 'entailment' if avg_probs[ENTAILMENT_INDEX] >= avg_probs[CONTRADICTION_INDEX] else 'contradiction'

    return strongest_label


def parse_rank_order(text, original_indices):
    match = re.search(r'<answer>(.*?)<\/answer>', text, re.DOTALL)
    if match:
        content = match.group(1)
        local_ids = re.findall(r'\[(\d+)\]', content)
        return [original_indices[int(i) - 1] for i in local_ids if i.isdigit() and int(i) <= len(original_indices)]
    return original_indices  # Fallback אם ה-Parsing נכשל


def perform_reasonrank_one_shot_with_window(claim_sentence, candidate_evidence_list, llm, window_size=20, step=1):
    all_indices = list(range(len(candidate_evidence_list)))

    if len(all_indices) <= window_size:
        current_ranked_indices = run_single_batch_indices(claim_sentence, candidate_evidence_list, all_indices, llm)
    else:
        current_ranked_indices = all_indices[:window_size]
        current_ranked_indices = run_single_batch_indices(claim_sentence, candidate_evidence_list, current_ranked_indices, llm)

        pointer = window_size
        while pointer < len(all_indices):
            next_batch = all_indices[pointer: pointer + step]
            combined_indices = current_ranked_indices[:(window_size - step)] + next_batch

            current_ranked_indices = run_single_batch_indices(claim_sentence, candidate_evidence_list, combined_indices, llm)
            pointer += step

    final_order = current_ranked_indices + [i for i in all_indices if i not in current_ranked_indices]

    ranked_evidence_dict = {}
    evidence_list = []
    for i, idx in enumerate(final_order, start=1):
        evidence_list.append(candidate_evidence_list[idx])
        ranked_evidence_dict[i] = {SELECTED_SNIPPETS: evidence_list.copy()}

    return ranked_evidence_dict


def run_single_batch_indices(claim_sentence, candidate_evidence_list, indices_to_rank, llm):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

    batch_texts = [candidate_evidence_list[i] for i in indices_to_rank]

    prompt = create_reasonrank_prompt(claim_sentence, batch_texts)

    outputs = llm.generate([prompt], sampling_params)
    raw_text = outputs[0].outputs[0].text
    return parse_rank_order(raw_text, indices_to_rank)


def create_reasonrank_prompt(claim_sentence, batch_texts):
    num = len(batch_texts)
    sentences_formatted = "\n".join([f"[{i + 1}]: {s}" for i, s in enumerate(batch_texts)])

    prompt = REASONRANK_PROMPT.format(num=num, claim_sentence=claim_sentence, sentences_formatted=sentences_formatted)

    return prompt


def perform_laquer_selection(claim_sentence, candidate_evidence_list):
    ranked_evidence_dict = {}

    prompt = create_prompt_for_evidence_selection_lauqer(claim_sentence, candidate_evidence_list)
    messages_list = create_messages_for_query(prompt)

    llm_raw_response, is_query_failed = query_openai_model(messages_list)

    if is_query_failed:
        ranked_evidence_dict[1] = {SELECTED_SNIPPETS: []}
    else:
        evidence_numbers_list = parse_llm_evidence_ranking_response(llm_raw_response)
        evidence_list = [candidate_evidence_list[i - 1] for i in evidence_numbers_list if
                         1 <= i <= len(candidate_evidence_list)]
        ranked_evidence_dict[1] = {SELECTED_SNIPPETS: evidence_list.copy()}

    return ranked_evidence_dict


def create_prompt_for_evidence_selection_lauqer(claim_sentence, candidate_evidence_list):
    prompt_pattern = LLM_EVIDENCE_SELECTION_PROMPT

    formatted_sentences = "\n".join(
        f"{i+1}. {sent}" for i, sent in enumerate(candidate_evidence_list)
    )

    prompt = prompt_pattern.format(
        NUMBER_OF_SENTENCES=len(candidate_evidence_list),
        claim=claim_sentence,
        sentences=formatted_sentences
    )

    return prompt


def update_instances_json_with_ranked_evidence(instances_with_ranked_evidence, ranked_evidence_dict, instance_id):
    instances_with_ranked_evidence[instance_id] = {}

    for sentence, supporting_citations_dict in ranked_evidence_dict.items():
        ranked_evidence_dict_for_json = {}

        for ranking_criterion, ranked_evidence_for_criterion_dict in ranked_evidence_dict.items():
            if ranking_criterion != GOLD and ranking_criterion != LABEL:
                ranked_evidence_for_criterion_list = ranked_evidence_for_criterion_dict[SELECTED_SNIPPETS]
                ranked_evidence_dict_for_json[ranking_criterion] = {SELECTED_SNIPPETS: ranked_evidence_for_criterion_list}
            elif ranking_criterion == GOLD:
                ranked_evidence_dict_for_json[GOLD] = ranked_evidence_for_criterion_dict
            else:
                ranked_evidence_dict_for_json[LABEL] = ranked_evidence_for_criterion_dict

        instances_with_ranked_evidence[instance_id][sentence] = ranked_evidence_dict_for_json


def get_top_ranked_gold_snippets_for_instance(claim_to_candidate_evidence_mapping_dict):
    top_ranked_gold_snippets_dict = {}

    for claim_sentence, evidence_candidates in claim_to_candidate_evidence_mapping_dict.items():
        gold_snippets_groups = evidence_candidates.get(GOLD).get(GOLD_SNIPPETS_GROUPS)
        optimal_gold_snippets = get_optimal_gold_snippets(gold_snippets_groups)

        selected_snippets_dict = {}
        selected_snippets_list = []

        for idx, snippet in enumerate(optimal_gold_snippets, start=1):
            selected_snippets_list.append(snippet)
            selected_snippets_dict[str(idx)] = {SELECTED_SNIPPETS: selected_snippets_list.copy()}

        candidate_snippets_list = evidence_candidates.get(CANDIDATE_SNIPPETS)
        non_optimal_gold_snippets = sorted(list((Counter(candidate_snippets_list) - Counter(optimal_gold_snippets)).elements()), key=lambda sent: len(sent.split()))
        selected_snippets_list = optimal_gold_snippets.copy()

        for idx, snippet in enumerate(non_optimal_gold_snippets, start=len(optimal_gold_snippets)+1):
            selected_snippets_list.append(snippet)
            selected_snippets_dict[str(idx)] = {SELECTED_SNIPPETS: selected_snippets_list.copy()}

        selected_snippets_dict[GOLD] = evidence_candidates.get(GOLD)
        selected_snippets_dict[LABEL] = evidence_candidates.get(LABEL)

        top_ranked_gold_snippets_dict[claim_sentence] = copy.deepcopy(selected_snippets_dict)

    return top_ranked_gold_snippets_dict


def get_optimal_gold_snippets(gold_snippets_groups):
    optimal_gold_snippets = []
    seen = set()

    gold_snippets_groups_sorted = sorted(gold_snippets_groups,
                                         key=lambda group: (len(group), sum(len(sent.split()) for sent in group)))

    for group in gold_snippets_groups_sorted:
        sorted_group = sorted(group, key=lambda sent: len(sent.split()))
        for sent in sorted_group:
            if sent not in seen:
                optimal_gold_snippets.append(sent)
                seen.add(sent)

    return optimal_gold_snippets


if __name__ == "__main__":
    with open(CLAIM_TO_CANDIDATE_EVIDENCE_MAPPING_PATH, "r", encoding="utf-8") as f:
        claim_to_candidate_evidence_mapping_instances = json.load(f)

    should_create_gold_ranking_json = True

    for evidence_ranking_method in EVIDENCE_RANKINGS_METHODS:
        instances_with_ranked_evidence, instances_with_ranked_gold_snippets = perform_citation_selection(
            claim_to_candidate_evidence_mapping_instances, evidence_ranking_method, should_create_gold_ranking_json)

        os.makedirs(INSTANCES_WITH_RANKED_EVIDENCE_PATH, exist_ok=True)
        filename = f"{INSTANCES_WITH_RANKED_EVIDENCE_PATH}{evidence_ranking_method}_ranked.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(instances_with_ranked_evidence, f, ensure_ascii=False, indent=4)

        if should_create_gold_ranking_json and SHOULD_USE_GOLD_SNIPPETS:
            with open(f"{INSTANCES_WITH_RANKED_EVIDENCE_PATH}{GOLD}_ranked.json", "w", encoding="utf-8") as f:
                json.dump(instances_with_ranked_gold_snippets, f, ensure_ascii=False, indent=4)

            should_create_gold_ranking_json = False
