#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Question Answerer
====================

This script takes questions from the dataset and asks GPT-4o-mini to answer them.
Forces JSON responses with only a, b, c, or d answers.
"""

import pandas as pd
import openai
import json
import os
import time
import random
from dotenv import load_dotenv
from tqdm import tqdm
import sys
import io
from datetime import datetime

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

class QuestionAnswerer:
    def __init__(self):
        """Initialize the question answerer with OpenAI API"""
        self.client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.model = "gpt-4.1-nano-2025-04-14"  # Using GPT-4o-mini model
        self.results_csv = f"llm_test_results_{self.model.replace('.', '_').replace('-', '_')}.csv"
        self.system_prompt = """Respond in JSON format: {"answer": "A"} where A/B/C/D is your choice."""
        self.initialize_results_csv()
    
    def initialize_results_csv(self):
        """Initialize the results CSV file if it doesn't exist"""
        if not os.path.exists(self.results_csv):
            # Create CSV with headers
            columns = [
                'question_id',
                'correct_answer', 
                'llm_answer',
                'model_name',
                'timestamp',
                'is_correct'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.results_csv, index=False, encoding='utf-8')
            print(f"✅ Created results CSV: {self.results_csv}")
        else:
            print(f"✅ Using existing results CSV: {self.results_csv}")
    
    def get_processed_question_ids(self):
        """Get list of already processed question IDs"""
        if os.path.exists(self.results_csv):
            try:
                df = pd.read_csv(self.results_csv, encoding='utf-8')
                if not df.empty:
                    processed_ids = set(df['question_id'].astype(str))
                    print(f"📋 Found {len(processed_ids)} already processed questions")
                    return processed_ids
            except Exception as e:
                print(f"⚠️ Error reading existing results: {e}")
        return set()
    
    def save_result_to_csv(self, question_id, correct_answer, llm_answer):
        """Save a single result to CSV immediately"""
        is_correct = llm_answer == correct_answer.upper() if llm_answer else False
        
        new_row = {
            'question_id': question_id,
            'correct_answer': correct_answer,
            'llm_answer': llm_answer,
            'model_name': self.model,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_correct': is_correct
        }
        
        # Append to CSV
        df = pd.DataFrame([new_row])
        df.to_csv(self.results_csv, mode='a', header=False, index=False, encoding='utf-8')
        
    def load_dataset(self, file_path):
        """Load the questions dataset"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"✅ Loaded dataset with {len(df):,} questions")
            return df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def parse_options(self, options_str):
        """Parse the options string into a list"""
        try:
            # Remove brackets and quotes, then split
            options_str = options_str.strip("[]")
            options = [opt.strip().strip("'\"") for opt in options_str.split("',")]
            # Clean up any remaining quotes
            options = [opt.replace("'", "").replace('"', '') for opt in options]
            return options
        except Exception as e:
            print(f"⚠️ Error parsing options: {e}")
            return []
    
    def format_question_for_llm(self, question, options):
        """Format question and options for the LLM using minimal prompt format"""
        # Ensure we have exactly 4 options
        if len(options) < 4:
            options.extend([''] * (4 - len(options)))
        
        formatted_question = f"""Answer this multiple-choice question. Respond only with A, B, C, or D.

{question}

A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}"""
        
        return formatted_question
    
    def ask_llm(self, question, options, max_retries=3):
        """Ask the LLM to answer the question"""
        formatted_question = self.format_question_for_llm(question, options)
        
        # No need to store full prompt anymore

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": formatted_question}
                    ],
                    # temperature=0.1,  # Low temperature for consistent answers
                    # max_tokens=50,    # Very short response
                    response_format={"type": "json_object"}  # Force JSON response
                )
                
                # Parse the JSON response
                response_text = response.choices[0].message.content.strip()
                response_json = json.loads(response_text)
                
                # Validate the response
                if "answer" in response_json:
                    answer = response_json["answer"].upper().strip()
                    if answer in ['A', 'B', 'C', 'D']:
                        return answer
                    else:
                        print(f"⚠️ Invalid answer format: {answer}")
                
                print(f"⚠️ Invalid response format: {response_text}")
                
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decode error (attempt {attempt + 1}): {e}")
            except Exception as e:
                print(f"⚠️ API error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        print(f"❌ Failed to get valid response after {max_retries} attempts")
        return None
    
    def test_single_question(self, df, index=None, show_details=True, skip_processed=True):
        """Test a single question"""
        if index is None:
            index = random.randint(0, len(df) - 1)
        
        row = df.iloc[index]
        question_id = row['Unique_Serial']
        
        # Check if already processed
        if skip_processed:
            processed_ids = self.get_processed_question_ids()
            if str(question_id) in processed_ids:
                if show_details:
                    print(f"⏭️ Question {question_id} already processed, skipping...")
                return None
        
        question = row['question']
        correct_answer = row['correct_answer']
        options_str = row['options']
        subject = row['subject']
        subdomain = row['subdomain_name']
        
        if show_details:
            print(f"\n{'='*60}")
            print(f"📝 TESTING QUESTION #{index + 1}")
            print(f"{'='*60}")
            print(f"Subject: {subject}")
            print(f"Subdomain: {subdomain}")
            print(f"Question: {question}")
        
        # Parse options
        options = self.parse_options(options_str)
        if len(options) != 4:
            if show_details:
                print(f"❌ Invalid options count: {len(options)}")
            return None
        
        if show_details:
            print(f"\nOptions:")
            for i, option in enumerate(options):
                letter = chr(65 + i)  # A, B, C, D
                print(f"  {letter}) {option}")
            
            print(f"\nCorrect Answer: {correct_answer}")
            print(f"\n🤖 Asking GPT-4o-mini...")
        
        # Ask LLM
        llm_answer = self.ask_llm(question, options)
        
        # Save result to CSV immediately
        self.save_result_to_csv(question_id, correct_answer, llm_answer)
        
        if llm_answer:
            is_correct = llm_answer == correct_answer.upper()
            if show_details:
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                print(f"LLM Answer: {llm_answer}")
                print(f"Result: {status}")
                print(f"💾 Saved to CSV: {self.results_csv}")
            
            return {
                'index': index,
                'question': question,
                'correct_answer': correct_answer,
                'llm_answer': llm_answer,
                'is_correct': is_correct,
                'subject': subject,
                'subdomain': subdomain,
                'question_id': question_id
            }
        else:
            if show_details:
                print(f"❌ Failed to get LLM response")
                print(f"💾 Saved failure to CSV: {self.results_csv}")
            return None
    
    def test_multiple_questions(self, df, num_questions=10, random_selection=True, resume=True):
        """Test multiple questions and return results"""
        results = []
        
        # Get already processed questions if resuming
        processed_ids = set()
        if resume:
            processed_ids = self.get_processed_question_ids()
        
        if random_selection:
            # Filter out already processed questions
            available_indices = []
            for i in range(len(df)):
                question_id = str(df.iloc[i]['Unique_Serial'])
                if not resume or question_id not in processed_ids:
                    available_indices.append(i)
            
            if len(available_indices) == 0:
                print("✅ All questions already processed!")
                return results
            
            indices = random.sample(available_indices, min(num_questions, len(available_indices)))
        else:
            # Sequential processing, skip already processed
            indices = []
            for i in range(len(df)):
                if len(indices) >= num_questions:
                    break
                question_id = str(df.iloc[i]['Unique_Serial'])
                if not resume or question_id not in processed_ids:
                    indices.append(i)
        
        if not indices:
            print("✅ All requested questions already processed!")
            return results
        
        remaining = len(indices)
        print(f"\n🚀 TESTING {remaining} QUESTIONS")
        if resume and processed_ids:
            print(f"📋 Resuming: {len(processed_ids)} already processed, {remaining} remaining")
        print(f"{'='*60}")
        
        # Use tqdm for progress bar
        for index in tqdm(indices, desc="Testing questions"):
            result = self.test_single_question(df, index, show_details=False, skip_processed=False)
            if result:
                results.append(result)
            
            # Optimized delay for Tier 1 (500 RPM = ~0.12s per request)
            time.sleep(0.15)
        
        return results
    
    def process_all_questions(self, df, start_from_beginning=False):
        """Process all questions in the dataset, resuming from where left off"""
        processed_ids = set()
        if not start_from_beginning:
            processed_ids = self.get_processed_question_ids()
        
        # Find unprocessed questions
        unprocessed_indices = []
        for i in range(len(df)):
            question_id = str(df.iloc[i]['Unique_Serial'])
            if start_from_beginning or question_id not in processed_ids:
                unprocessed_indices.append(i)
        
        if not unprocessed_indices:
            print("✅ All questions already processed!")
            return []
        
        total_questions = len(df)
        remaining = len(unprocessed_indices)
        processed_count = total_questions - remaining
        
        print(f"\n🚀 PROCESSING ALL QUESTIONS")
        print(f"{'='*60}")
        print(f"Total questions: {total_questions:,}")
        print(f"Already processed: {processed_count:,}")
        print(f"Remaining: {remaining:,}")
        print(f"{'='*60}")
        
        results = []
        
        # Process with progress bar
        for index in tqdm(unprocessed_indices, desc="Processing all questions"):
            result = self.test_single_question(df, index, show_details=False, skip_processed=False)
            if result:
                results.append(result)
            
            # Optimized delay for Tier 1 (500 RPM = ~0.12s per request)
            time.sleep(0.15)
        
        return results
    
    def analyze_results(self, results):
        """Analyze and display results"""
        if not results:
            print("❌ No results to analyze")
            return
        
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r['is_correct'])
        accuracy = (correct_answers / total_questions) * 100
        
        print(f"\n{'='*60}")
        print(f"📊 RESULTS ANALYSIS")
        print(f"{'='*60}")
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {correct_answers}")
        print(f"Incorrect Answers: {total_questions - correct_answers}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"💾 All results saved to: {self.results_csv}")
        
        # Analyze by subject
        subject_stats = {}
        for result in results:
            subject = result['subject']
            if subject not in subject_stats:
                subject_stats[subject] = {'total': 0, 'correct': 0}
            subject_stats[subject]['total'] += 1
            if result['is_correct']:
                subject_stats[subject]['correct'] += 1
        
        print(f"\n📚 ACCURACY BY SUBJECT:")
        for subject, stats in subject_stats.items():
            subject_accuracy = (stats['correct'] / stats['total']) * 100
            print(f"  {subject}: {stats['correct']}/{stats['total']} ({subject_accuracy:.1f}%)")
        
        # Show incorrect answers
        incorrect_results = [r for r in results if not r['is_correct']]
        if incorrect_results:
            print(f"\n❌ INCORRECT ANSWERS:")
            for result in incorrect_results:
                print(f"  Q{result['index'] + 1}: Expected {result['correct_answer']}, Got {result['llm_answer']} - {result['subject']}")
        
        return {
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'accuracy': accuracy,
            'subject_stats': subject_stats,
            'results': results
        }
    
    def save_results(self, results, filename='llm_test_results.json'):
        """Save results to a JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main function"""
    print("🤖 LLM Question Answerer")
    print("=" * 40)
    
    # Initialize the answerer
    answerer = QuestionAnswerer()
    
    # Load dataset
    df = answerer.load_dataset("merged_all_questions_with_subdomains_renamed.csv")
    if df is None:
        return
    
    # Test options
    print(f"\nChoose an option:")
    print(f"1. Test a single random question")
    print(f"2. Test multiple questions (specify number)")
    print(f"3. Test a specific question by index")
    print(f"4. Process ALL questions (resume from where left off)")
    print(f"5. Process ALL questions (start from beginning)")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Test single random question
            result = answerer.test_single_question(df)
            if result:
                answerer.analyze_results([result])
        
        elif choice == '2':
            # Test multiple questions
            num_questions = int(input("How many questions to test? "))
            results = answerer.test_multiple_questions(df, num_questions)
            analysis = answerer.analyze_results(results)
            if analysis:
                answerer.save_results(analysis)
        
        elif choice == '3':
            # Test specific question
            index = int(input(f"Enter question index (0-{len(df)-1}): "))
            if 0 <= index < len(df):
                result = answerer.test_single_question(df, index)
                if result:
                    answerer.analyze_results([result])
            else:
                print(f"❌ Invalid index. Must be between 0 and {len(df)-1}")
        
        elif choice == '4':
            # Process all questions (resume)
            print("🔄 Processing all questions, resuming from where left off...")
            results = answerer.process_all_questions(df, start_from_beginning=False)
            analysis = answerer.analyze_results(results)
            if analysis:
                answerer.save_results(analysis, 'final_analysis.json')
        
        elif choice == '5':
            # Process all questions (start over)
            confirm = input("⚠️ This will reprocess ALL questions. Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                print("🔄 Processing all questions from the beginning...")
                results = answerer.process_all_questions(df, start_from_beginning=True)
                analysis = answerer.analyze_results(results)
                if analysis:
                    answerer.save_results(analysis, 'final_analysis.json')
            else:
                print("❌ Cancelled")
        
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print(f"\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
