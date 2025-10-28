from CommonTools.credentials import GPT_API_KEY
from openai import OpenAI
from CommonTools.prompts import sentiment, de_neuter_name, mask_template, para_template, mask_template_iter, para_template_iter
from CommonTools.GPT_agent import GPT_Agent, Ollama_Agent
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import os
import datetime
import random
from CommonTools.CommonTools import _make_agent, process_llm_responses, process_llm_responses_dynamic, extract_sentiment_features, extract_it_features, check_if_identified
pd.options.mode.chained_assignment = None  # Disable chained assignment warning
output_dir = '/Users/User/Desktop/ParProcess_Output/'

# Define all neuter arguments as a global dict
NEUTER_ARGS = {
    'output_dir': output_dir,
    'gpt_api_key': GPT_API_KEY,
    'mask_template': mask_template,
    'para_template': para_template,
    'mask_template_iter': mask_template_iter,
    'para_template_iter': para_template_iter,
    'de_neuter_name': de_neuter_name,
    'sentiment_template': sentiment,
    # Optional parameters - customize as needed
    'max_rounds': 8,
    'model_name': 'gemma3:4b', # gpt-4o-mini # gpt-5-nano-2025-08-07 # gpt-oss:20b # gemma3:27b # gemma3:4b
    'save_intermediate': True,
    'save_round_files': True,
    'perform_sentiment': True,
    'sentiment_on_raw': True,
    'sentiment_on_neutered': True,
    'verbose': True,
    'log_errors': True,
    'error_log_path': None,
    'use_random_filename': True,
    'custom_filename_prefix': "instance",
    'return_dataframe': False,
    'llm_provider': 'ollama', #ollama # openai
    'paraphrase': True,
    'masking': True,
    'extraction_name': 'LLM_RAW'
}


def neuter_wrapper(instance_data):
    """Simple wrapper that unpacks kwargs."""
    return neuter_data(instance_data, **NEUTER_ARGS)


def neuter_data(
        instance_data,
        output_dir,
        gpt_api_key,
        mask_template,
        para_template,
        mask_template_iter,
        para_template_iter,
        de_neuter_name,
        sentiment_template,
        # Customizable parameters
        max_rounds=8,
        model_name='gpt-5-nano-2025-08-07',
        save_intermediate=True,
        save_round_files=True,
        perform_sentiment=True,
        sentiment_on_raw=True,
        sentiment_on_neutered=True,
        verbose=True,
        log_errors=True,
        error_log_path=None,
        use_random_filename=True,
        custom_filename_prefix="instance",
        return_dataframe=False,
        llm_provider='openai',
        paraphrase=True,
        masking=True,  # NEW: Parameter to control masking
        extraction_name = 'LLM_RAW'
):
    """
    Process text data through masking, paraphrasing, and sentiment analysis pipeline.
    """
    try:
        # Generate filename based on settings
        if use_random_filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            random_number = random.randint(1, 10000)
            base_filename = f"{custom_filename_prefix}_{instance_data.index[0]}_{timestamp}_{random_number}.csv"
        else:
            base_filename = f"{custom_filename_prefix}_{instance_data.index[0]}.csv"
            random_number = instance_data.index[0]

        output_path = os.path.join(output_dir, base_filename)

        if verbose:
            print(f"Processing instance {random_number}...")
            print(f"Output path: {output_path}")
            print(f"Masking enabled: {masking}")
            print(f"Paraphrasing enabled: {paraphrase}")

        if not masking and not paraphrase:
            if verbose:
                print(f"Instance {random_number}: Both masking and paraphrasing disabled - working with raw text only")

            # Skip all neutering steps, work directly with raw text
            random_sample = instance_data.copy()
            random_sample['Body Text Neutered'] = random_sample['Body Text']
            random_sample['neutered_at_round'] = 0
            random_sample['neutered_at_step'] = 'no_neutering'

            # Skip sentiment on neutered text if requested
            if sentiment_on_neutered:
                print(
                    f"WARNING Instance {random_number}: sentiment_on_neutered is True but masking and paraphrasing are False. Skipping neutered sentiment.")

            # Only perform sentiment on raw text if requested
            if perform_sentiment and sentiment_on_raw:
                if verbose:
                    print(f"Instance {random_number}: Performing sentiment analysis on raw text...")
                sentiment_agent = _make_agent(llm_provider, gpt_api_key, sentiment_template, model_name)
                random_sample = process_llm_responses(random_sample, extraction_name, 'Body Text', sentiment_agent, provider=llm_provider)
                #random_sample = extract_sentiment_features(random_sample, extraction_name, ['Direction', 'Magnitude'])

            # Save final result
            final_save_path = output_path.replace(".csv", "_FINAL_FINAL_FINAL.csv")
            random_sample.to_csv(final_save_path)

            if verbose:
                print(f"Instance {random_number}: Processing completed (raw text only).")
                print(f"Instance {random_number}: Final output saved to {final_save_path}")

            if return_dataframe:
                return random_sample
            return

        # Initialize list to track all processed texts
        all_neutered_texts = []

        # Step 1: Initial Masking (only if masking is enabled)
        if masking:
            if verbose:
                print(f"Instance {random_number}: Applying initial masking...")

            random_sample = process_llm_responses_dynamic(
                instance_data,
                'Body Text Masked',
                'Body Text',
                mask_template,
                model=model_name,
                provider=llm_provider
            )

            if save_intermediate:
                neuter_save_path = output_path.replace(".csv", "_step1_masked.csv")
                random_sample.to_csv(neuter_save_path)
                if verbose:
                    print(f"  Saved intermediate: {neuter_save_path}")

            # Check identification after masking
            if verbose:
                print(f"Instance {random_number}: Checking identification after masking...")

            deneuter_agent = _make_agent(llm_provider, gpt_api_key, de_neuter_name, model_name)
            random_sample = process_llm_responses(random_sample, 'LLM_DENEUTER_GUESS_MASK', 'Body Text Masked',
                                                  deneuter_agent, provider=llm_provider)
            random_sample = extract_it_features(
                random_sample,
                'LLM_DENEUTER_GUESS_MASK',
                ['Ticker_Guess', 'Name_Guess', 'Date_Guess', 'Industry_Guess']
            )
            random_sample = check_if_identified(random_sample, suffix='_masking_step_0')

            # Separate neutered and identified after masking
            neutered_after_mask = random_sample[random_sample['iort_guess_correct_masking_step_0'] == 0].copy()
            identified_after_mask = random_sample[random_sample['iort_guess_correct_masking_step_0'] == 1].copy()

            if not neutered_after_mask.empty:
                neutered_after_mask['neutered_at_round'] = 0
                neutered_after_mask['neutered_at_step'] = 'masking_step_0'
                neutered_after_mask['Body Text Neutered'] = neutered_after_mask['Body Text Masked']
                all_neutered_texts.append(neutered_after_mask)

                if save_round_files:
                    save_path = output_path.replace(".csv", "_masking_step_0_neutered.csv")
                    neutered_after_mask.to_csv(save_path)
                    if verbose:
                        print(f"  {len(neutered_after_mask)} texts neutered after masking, saved to {save_path}")
        else:
            # Skip masking, start with raw text
            if verbose:
                print(f"Instance {random_number}: Masking disabled, skipping masking step...")
            random_sample = instance_data.copy()
            identified_after_mask = random_sample.copy()

        # Step 2: Paraphrase if enabled and there are still identified texts (or if we skipped masking)
        if paraphrase and not identified_after_mask.empty:
            if verbose:
                print(f"Instance {random_number}: Applying paraphrasing to {len(identified_after_mask)} texts...")

            # Use appropriate source column based on whether masking was done
            source_col = 'Body Text Masked' if masking else 'Body Text'
            identified_after_mask = process_llm_responses_dynamic(
                identified_after_mask,
                'Body Text Paraphrased',
                source_col,
                para_template,
                model=model_name,
                provider=llm_provider
            )

            if save_intermediate:
                neuter_save_path = output_path.replace(".csv", "_step2_paraphrased.csv")
                identified_after_mask.to_csv(neuter_save_path)
                if verbose:
                    print(f"  Saved intermediate: {neuter_save_path}")

            # Check identification after paraphrasing
            if verbose:
                print(f"Instance {random_number}: Checking identification after paraphrasing...")

            deneuter_agent = _make_agent(llm_provider, gpt_api_key, de_neuter_name, model_name)
            identified_after_mask = process_llm_responses(identified_after_mask, 'LLM_DENEUTER_GUESS_PARA',
                                                          'Body Text Paraphrased', deneuter_agent,
                                                          provider=llm_provider)
            identified_after_mask = extract_it_features(
                identified_after_mask,
                'LLM_DENEUTER_GUESS_PARA',
                ['Ticker_Guess', 'Name_Guess', 'Date_Guess', 'Industry_Guess']
            )
            identified_after_mask = check_if_identified(identified_after_mask, suffix='_paraphrasing_step_0')

            # Separate neutered and identified after paraphrasing
            neutered_after_para = identified_after_mask[
                identified_after_mask['iort_guess_correct_paraphrasing_step_0'] == 0].copy()
            still_identified = identified_after_mask[
                identified_after_mask['iort_guess_correct_paraphrasing_step_0'] == 1].copy()

            if not neutered_after_para.empty:
                neutered_after_para['neutered_at_round'] = 0
                neutered_after_para['neutered_at_step'] = 'paraphrasing_step_0'
                neutered_after_para['Body Text Neutered'] = neutered_after_para['Body Text Paraphrased']
                all_neutered_texts.append(neutered_after_para)

                if save_round_files:
                    save_path = output_path.replace(".csv", "_paraphrasing_step_0_neutered.csv")
                    neutered_after_para.to_csv(save_path)
                    if verbose:
                        print(f"  {len(neutered_after_para)} texts neutered after paraphrasing, saved to {save_path}")

            # Update for iteration
            texts_to_process = still_identified.copy()
            if not texts_to_process.empty:
                texts_to_process['Body Text Neutered'] = texts_to_process['Body Text Paraphrased']

        elif not paraphrase and not identified_after_mask.empty and masking:
            # If paraphrasing is disabled but masking was done, move identified texts directly to iteration
            texts_to_process = identified_after_mask.copy()
            texts_to_process['Body Text Neutered'] = texts_to_process['Body Text Masked']
            if verbose:
                print(
                    f"Instance {random_number}: Skipping paraphrasing, {len(texts_to_process)} texts remain identified...")
        else:
            texts_to_process = pd.DataFrame()

        # ITERATIVE LOOP (only if at least one of masking or paraphrasing is enabled)
        if not texts_to_process.empty and (masking or paraphrase):
            if verbose:
                print(f"Instance {random_number}: Starting iterative process with {len(texts_to_process)} texts...")

            for round_num in range(1, max_rounds + 1):
                if verbose:
                    print(f"Instance {random_number}: Starting round {round_num} with {len(texts_to_process)} texts...")

                if texts_to_process.empty:
                    if verbose:
                        print(f"Instance {random_number}: No more texts to process.")
                    break

                # Iteration Step 1: Additional masking (only if masking is enabled)
                if masking:
                    texts_to_process = process_llm_responses_dynamic(
                        texts_to_process,
                        'Body Text Masked Iter',
                        'Body Text Neutered',
                        mask_template_iter,
                        model=model_name,
                        provider=llm_provider
                    )

                    if save_intermediate and save_round_files:
                        neuter_save_path = output_path.replace(".csv", f"_round_{round_num}_masked.csv")
                        texts_to_process.to_csv(neuter_save_path)

                    # Check identification after iterative masking
                    if verbose:
                        print(f"  Round {round_num}: Checking identification after masking...")

                    deneuter_agent = _make_agent(llm_provider, gpt_api_key, de_neuter_name, model_name)
                    texts_to_process = process_llm_responses(
                        texts_to_process,
                        f'LLM_DENEUTER_GUESS_MASK_R{round_num}',
                        'Body Text Masked Iter',
                        deneuter_agent, provider=llm_provider
                    )
                    texts_to_process = extract_it_features(
                        texts_to_process,
                        f'LLM_DENEUTER_GUESS_MASK_R{round_num}',
                        ['Ticker_Guess', 'Name_Guess', 'Date_Guess', 'Industry_Guess']
                    )
                    texts_to_process = check_if_identified(texts_to_process, suffix=f'_masking_step_{round_num}')

                    # Check for newly neutered texts after masking
                    newly_neutered_mask = texts_to_process[
                        texts_to_process[f'iort_guess_correct_masking_step_{round_num}'] == 0].copy()
                    still_identified_mask = texts_to_process[
                        texts_to_process[f'iort_guess_correct_masking_step_{round_num}'] == 1].copy()

                    if not newly_neutered_mask.empty:
                        newly_neutered_mask['neutered_at_round'] = round_num
                        newly_neutered_mask['neutered_at_step'] = f'masking_step_{round_num}'
                        newly_neutered_mask['Body Text Neutered'] = newly_neutered_mask['Body Text Masked Iter']
                        all_neutered_texts.append(newly_neutered_mask)

                        if save_round_files:
                            save_path = output_path.replace(".csv", f"_masking_step_{round_num}_neutered.csv")
                            newly_neutered_mask.to_csv(save_path)

                        if verbose:
                            print(f"    {len(newly_neutered_mask)} texts neutered after masking")
                else:
                    # Skip masking in iteration
                    still_identified_mask = texts_to_process.copy()
                    if verbose:
                        print(f"  Round {round_num}: Skipping masking step...")

                # Continue with paraphrasing if enabled and texts remain
                if paraphrase and not still_identified_mask.empty:
                    # Use appropriate source column based on whether masking was done
                    source_col = 'Body Text Masked Iter' if masking else 'Body Text Neutered'
                    still_identified_mask = process_llm_responses_dynamic(
                        still_identified_mask,
                        'Body Text Paraphrased Iter',
                        source_col,
                        para_template_iter,
                        model=model_name,
                        provider=llm_provider
                    )

                    if save_intermediate and save_round_files:
                        neuter_save_path = output_path.replace(".csv", f"_round_{round_num}_paraphrased.csv")
                        still_identified_mask.to_csv(neuter_save_path)

                    # Check identification after iterative paraphrasing
                    if verbose:
                        print(f"  Round {round_num}: Checking identification after paraphrasing...")

                    deneuter_agent = _make_agent(llm_provider, gpt_api_key, de_neuter_name, model_name)
                    still_identified_mask = process_llm_responses(
                        still_identified_mask,
                        f'LLM_DENEUTER_GUESS_PARA_R{round_num}',
                        'Body Text Paraphrased Iter',
                        deneuter_agent, provider=llm_provider
                    )
                    still_identified_mask = extract_it_features(
                        still_identified_mask,
                        f'LLM_DENEUTER_GUESS_PARA_R{round_num}',
                        ['Ticker_Guess', 'Name_Guess', 'Date_Guess', 'Industry_Guess']
                    )
                    still_identified_mask = check_if_identified(still_identified_mask,
                                                                suffix=f'_paraphrasing_step_{round_num}')

                    # Check for newly neutered texts after paraphrasing
                    newly_neutered_para = still_identified_mask[
                        still_identified_mask[f'iort_guess_correct_paraphrasing_step_{round_num}'] == 0].copy()
                    still_identified = still_identified_mask[
                        still_identified_mask[f'iort_guess_correct_paraphrasing_step_{round_num}'] == 1].copy()

                    if not newly_neutered_para.empty:
                        newly_neutered_para['neutered_at_round'] = round_num
                        newly_neutered_para['neutered_at_step'] = f'paraphrasing_step_{round_num}'
                        newly_neutered_para['Body Text Neutered'] = newly_neutered_para['Body Text Paraphrased Iter']
                        all_neutered_texts.append(newly_neutered_para)

                        if save_round_files:
                            save_path = output_path.replace(".csv", f"_paraphrasing_step_{round_num}_neutered.csv")
                            newly_neutered_para.to_csv(save_path)

                        if verbose:
                            print(f"    {len(newly_neutered_para)} texts neutered after paraphrasing")

                    # Update for next iteration
                    texts_to_process = still_identified.copy()
                    if not texts_to_process.empty:
                        texts_to_process['Body Text Neutered'] = texts_to_process['Body Text Paraphrased Iter']
                else:
                    # If paraphrasing disabled, update for next iteration
                    texts_to_process = still_identified_mask.copy()
                    if not texts_to_process.empty and masking:
                        texts_to_process['Body Text Neutered'] = texts_to_process['Body Text Masked Iter']
                        if verbose:
                            print(f"  Round {round_num}: Skipping paraphrasing step...")

                # Check termination
                if texts_to_process.empty:
                    if verbose:
                        print(f"Instance {random_number}: All texts successfully neutered by round {round_num}")
                    break
                elif round_num == max_rounds:
                    # Mark texts that couldn't be neutered
                    texts_to_process['neutered_at_round'] = -1
                    texts_to_process['neutered_at_step'] = 'never_neutered'
                    all_neutered_texts.append(texts_to_process)
                    if verbose:
                        print(
                            f"Instance {random_number}: Reached max rounds. {len(texts_to_process)} texts remain identified.")
                    break

        # Combine all results
        if all_neutered_texts:
            final_combined = pd.concat(all_neutered_texts, ignore_index=True)
        else:
            final_combined = pd.DataFrame()

        if final_combined.empty:
            if verbose:
                print(f"Instance {random_number}: No texts to process.")
            if return_dataframe:
                return final_combined
            return

        if verbose:
            print(f"Instance {random_number}: Creating final identification columns from neutering step results...")

        # Initialize the final guess columns
        final_combined['Ticker_Guess_Neutered'] = ''
        final_combined['Name_Guess_Neutered'] = ''
        final_combined['Date_Guess_Neutered'] = ''
        final_combined['Industry_Guess_Neutered'] = ''

        # For each row, copy the guess values from the step where it was neutered
        for idx, row in final_combined.iterrows():
            neutered_step = row.get('neutered_at_step', '')

            if neutered_step == 'masking_step_0':
                if 'Ticker_Guess' in row and pd.notna(row['Ticker_Guess']):
                    final_combined.at[idx, 'Ticker_Guess_Neutered'] = row['Ticker_Guess']
                if 'Name_Guess' in row and pd.notna(row['Name_Guess']):
                    final_combined.at[idx, 'Name_Guess_Neutered'] = row['Name_Guess']
                if 'Date_Guess' in row and pd.notna(row['Date_Guess']):
                    final_combined.at[idx, 'Date_Guess_Neutered'] = row['Date_Guess']
                if 'Industry_Guess' in row and pd.notna(row['Industry_Guess']):
                    final_combined.at[idx, 'Industry_Guess_Neutered'] = row['Industry_Guess']

            elif neutered_step == 'paraphrasing_step_0':
                if 'Ticker_Guess' in row and pd.notna(row['Ticker_Guess']):
                    final_combined.at[idx, 'Ticker_Guess_Neutered'] = row['Ticker_Guess']
                if 'Name_Guess' in row and pd.notna(row['Name_Guess']):
                    final_combined.at[idx, 'Name_Guess_Neutered'] = row['Name_Guess']
                if 'Date_Guess' in row and pd.notna(row['Date_Guess']):
                    final_combined.at[idx, 'Date_Guess_Neutered'] = row['Date_Guess']
                if 'Industry_Guess' in row and pd.notna(row['Industry_Guess']):
                    final_combined.at[idx, 'Industry_Guess_Neutered'] = row['Industry_Guess']

            elif 'masking_step_' in neutered_step:
                round_num = neutered_step.split('_')[-1]
                ticker_col = f'Ticker_Guess'
                name_col = f'Name_Guess'
                date_col = f'Date_Guess'
                industry_col = f'Industry_Guess'

                if ticker_col in row and pd.notna(row[ticker_col]):
                    final_combined.at[idx, 'Ticker_Guess_Neutered'] = row[ticker_col]
                if name_col in row and pd.notna(row[name_col]):
                    final_combined.at[idx, 'Name_Guess_Neutered'] = row[name_col]
                if date_col in row and pd.notna(row[date_col]):
                    final_combined.at[idx, 'Date_Guess_Neutered'] = row[date_col]
                if industry_col in row and pd.notna(row[industry_col]):
                    final_combined.at[idx, 'Industry_Guess_Neutered'] = row[industry_col]

            elif 'paraphrasing_step_' in neutered_step:
                round_num = neutered_step.split('_')[-1]
                ticker_col = f'Ticker_Guess'
                name_col = f'Name_Guess'
                date_col = f'Date_Guess'
                industry_col = f'Industry_Guess'

                if ticker_col in row and pd.notna(row[ticker_col]):
                    final_combined.at[idx, 'Ticker_Guess_Neutered'] = row[ticker_col]
                if name_col in row and pd.notna(row[name_col]):
                    final_combined.at[idx, 'Name_Guess_Neutered'] = row[name_col]
                if date_col in row and pd.notna(row[date_col]):
                    final_combined.at[idx, 'Date_Guess_Neutered'] = row[date_col]
                if industry_col in row and pd.notna(row[industry_col]):
                    final_combined.at[idx, 'Industry_Guess_Neutered'] = row[industry_col]

            elif neutered_step == 'never_neutered':
                if 'Ticker_Guess' in row and pd.notna(row['Ticker_Guess']):
                    final_combined.at[idx, 'Ticker_Guess_Neutered'] = row['Ticker_Guess']
                if 'Name_Guess' in row and pd.notna(row['Name_Guess']):
                    final_combined.at[idx, 'Name_Guess_Neutered'] = row['Name_Guess']
                if 'Date_Guess' in row and pd.notna(row['Date_Guess']):
                    final_combined.at[idx, 'Date_Guess_Neutered'] = row['Date_Guess']
                if 'Industry_Guess' in row and pd.notna(row['Industry_Guess']):
                    final_combined.at[idx, 'Industry_Guess_Neutered'] = row['Industry_Guess']

        # Now run check_if_identified using the assembled guess columns
        final_combined = check_if_identified(
            final_combined,
            date_col='Date_Guess_Neutered',
            name_col='Name_Guess_Neutered',
            suffix='_neutered'
        )

        if verbose:
            print(f"Instance {random_number}: Final identification columns created from neutering step results.")

        # Save before sentiment analysis
        if not perform_sentiment:
            final_string = output_path.replace(".csv", "_FINAL_FINAL_FINAL.csv")
            final_combined.to_csv(final_string)
            if verbose:
                print(f"Instance {random_number}: Processing completed (no sentiment). Saved to {final_string}")
            if return_dataframe:
                return final_combined
            return

        # Save neutered texts before sentiment
        neutered_string = output_path.replace(".csv", "_NEUTERED.csv")
        final_combined.to_csv(neutered_string)
        if verbose:
            print(f"Instance {random_number}: Saved all neutered texts to {neutered_string}")

        # SENTIMENT ANALYSIS SECTION
        if sentiment_on_neutered:
            # NEW: Check if we actually have neutered text (not just raw text)
            if not masking and not paraphrase:
                print(
                    f"WARNING Instance {random_number}: sentiment_on_neutered is True but masking and paraphrasing are False. Skipping neutered sentiment.")
            else:
                if verbose:
                    print(f"Instance {random_number}: Performing sentiment analysis on neutered text...")

                sentiment_agent = _make_agent(llm_provider, gpt_api_key, sentiment_template, model_name)
                final_combined = process_llm_responses(final_combined, 'LLM_NEUT', 'Body Text Neutered',
                                                       sentiment_agent, provider=llm_provider)
                final_combined = extract_sentiment_features(final_combined, 'LLM_NEUT',
                                                            ['Direction_neut', 'Magnitude_neut'])

                if save_intermediate:
                    sentiment_neut_save_path = output_path.replace(".csv", "_sent_neut.csv")
                    final_combined.to_csv(sentiment_neut_save_path)
                    if verbose:
                        print(f"  Saved sentiment (neutered) analysis: {sentiment_neut_save_path}")

        if sentiment_on_raw:
            if verbose:
                print(f"Instance {random_number}: Performing sentiment analysis on raw text...")

            sentiment_agent = _make_agent(llm_provider, gpt_api_key, sentiment_template, model_name)
            final_combined = process_llm_responses(final_combined, extraction_name, 'Body Text', sentiment_agent,
                                                   provider=llm_provider)
            final_combined = extract_sentiment_features(final_combined, extraction_name, ['Direction', 'Magnitude'])

        # Save final result with the expected filename
        final_save_path = output_path.replace(".csv", "_FINAL_FINAL_FINAL.csv")
        final_combined.to_csv(final_save_path)

        if verbose:
            print(f"Instance {random_number}: Processing completed successfully.")
            print(f"Instance {random_number}: Final output saved to {final_save_path}")
            print(f"Instance {random_number}: Total texts processed: {len(final_combined)}")

            # Summary statistics if verbose
            if 'neutered_at_step' in final_combined.columns:
                step_stats = final_combined['neutered_at_step'].value_counts()
                print(f"Instance {random_number}: Neutering summary by step:")
                for step, count in step_stats.items():
                    print(f"  {step}: {count} texts")

        if return_dataframe:
            return final_combined

    except Exception as e:
        error_message = f"Error processing instance {random_number if 'random_number' in locals() else 'unknown'}: {e}"
        print(f"ERROR: {error_message}")

        if log_errors:
            if error_log_path is None:
                error_log_path = os.path.join(output_dir, 'error_log.txt')

            with open(error_log_path, 'a') as f:
                f.write(f"{datetime.datetime.now()} - {error_message}\n")
                import traceback
                f.write(f"Traceback: {traceback.format_exc()}\n")

        if return_dataframe:
            return None
        return


if __name__ == "__main__":
    mp.set_start_method("spawn")  # For macOS multiprocessing
    alldata = pd.read_csv('data.csv')

    alldata['Date'] = pd.to_datetime(alldata['Date'],format='mixed')
    print(alldata['COMNAM'].nunique())  # Must have a COMNAM column which is the company name
    print(alldata['Companies'].nunique())  # Must have a Companies column which is the company ticker
    print(alldata['SIC_Industry'].nunique())  # Must have a SIC_Industry which is the industry of the firm
    print(f"Earliest date: {alldata['Date'].min()}")
    print(f"Latest date: {alldata['Date'].max()}")

    random_sample = alldata.sample(1000) #random sample for speed and or cost savings

    # specs for multiprocessing
    os.makedirs(output_dir, exist_ok=True)
    num_workers = mp.cpu_count()
    num_splits = num_workers
    data_list = np.array_split(random_sample, num_splits)

    print("Starting LLM processing...")
    start_time_main = time.time()
    with mp.Pool(num_workers) as pool:
        pool.map(neuter_wrapper, data_list)  # Use the wrapper function
    end_time_main = time.time()
    optimized_time_main = end_time_main - start_time_main
    minutes = optimized_time_main / 60
    print("")
    print(f"Compiling Everything took: {minutes:.2f} minutes")
    print("")

    # List to store data from each CSV
    print('Starting to save into one file...')
    dataframes = []
    for filename in os.listdir(output_dir):
        if filename.endswith("_FINAL_FINAL_FINAL.csv"):
            file_path = os.path.join(output_dir, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)

    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, 'Finished_Data.csv'))
        print(f'Success! Combined {len(dataframes)} files.')
    else:
        print('Warning: No files found with pattern "_FINAL_FINAL_FINAL.csv"')

    print(f"The identification rate is {round(100*combined_df['iort_guess_correct_neutered'].mean(),3)}%")