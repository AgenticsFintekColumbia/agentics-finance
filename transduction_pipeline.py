"""
Simplified Transduction Pipeline
Direct transduction-based question answering without CrewAI or tool calling.
"""

import pandas as pd
import json
import asyncio
import os
import csv
import sys
import traceback
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from agentics import AG


class TransductionAnswer(BaseModel):
    """Final answer model for transduction."""
    detailed_answer: str = Field(
        ...,
        description="A comprehensive, detailed multi-paragraph answer with thorough evidence from the dataset. This should be at least 3-5 paragraphs long and include specific data points, trends, and insights."
    )
    explanation: str = Field(
        ...,
        description="A detailed reasoning and explanation of how the answer was derived from the data. Include the analytical process and key evidence used."
    )


class TransductionPipeline:
    """
    Simplified pipeline that always uses transduction to answer questions.
    No tool calling, no CrewAI - just pure transduction.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the transduction pipeline.

        Args:
            data_dir: Directory containing data files. If None, uses default data directory.
        """
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        else:
            self.data_dir = data_dir

        self.csv_path = None
        self.agentics = None

    def load_data(self, selected_columns: Optional[List[str]] = None) -> bool:
        """
        Load the merged dataset and optionally filter to selected columns.

        Args:
            selected_columns: List of column names to include in analysis

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Import the CSV reader utility
            from utils.csv_reader import get_merged_data_file_for_agentics

            try:
                self.csv_path = get_merged_data_file_for_agentics(self.data_dir)
            except FileNotFoundError as e:
                print(f"❌ ERROR: {e}")
                return False

            # Increase CSV field size limit to handle large text fields
            try:
                csv.field_size_limit(10 * 1024 * 1024)  # 10MB
            except OverflowError:
                csv.field_size_limit(sys.maxsize)

            # Load the full dataset
            print(f"\n📂 Loading merged dataset from {self.csv_path}")
            self.agentics = AG.from_csv(self.csv_path)
            print(f"✅ Loaded {len(self.agentics.states):,} total rows with {len(self.agentics.atype.model_fields)} columns")

            # Filter to selected columns if provided
            if selected_columns is not None and len(selected_columns) > 0:
                self.agentics = self._filter_to_columns(selected_columns)
                if self.agentics is None:
                    return False
            else:
                print(f"⚠️ No columns selected - using all columns")

            return True

        except Exception as e:
            print(f"❌ ERROR loading data: {e}")
            traceback.print_exc()
            return False

    def _filter_to_columns(self, selected_columns: List[str]) -> Optional[AG]:
        """
        Filter the dataset to only include selected columns.

        Args:
            selected_columns: List of column names to include

        Returns:
            AG object with filtered columns, or None if error
        """
        try:
            import re

            def sanitize_field_name(name: str) -> str:
                """Sanitize field name to match Agentics' sanitization."""
                name = name.strip()
                name = re.sub(r"^_+", "", name)
                if re.fullmatch(r"[a-zA-Z0-9_]+", name):
                    return name
                return re.sub(r"[^\w]", "", name)

            # Sanitize all selected columns
            sanitized_selected = [sanitize_field_name(col) for col in selected_columns]

            # Always include Date column for filtering
            selected_without_date = [col for col in sanitized_selected if col != "Date"]
            columns_to_include = ["Date"] + selected_without_date

            print(f"\n🔧 Filtering to {len(columns_to_include)} selected columns")
            print(f"   Columns: {', '.join(columns_to_include[:10])}{'...' if len(columns_to_include) > 10 else ''}")

            # Use Agentics __call__ method to filter columns
            filtered_agentics = self.agentics(*columns_to_include)
            print(f"✅ Filtered dataset to {len(columns_to_include)} columns (Date first)")

            return filtered_agentics

        except Exception as e:
            print(f"❌ ERROR filtering columns: {e}")
            traceback.print_exc()
            return None

    def _filter_by_date_range(self, agentics: AG, start_date: str, end_date: str) -> Optional[AG]:
        """
        Filter dataset to specified date range.

        Args:
            agentics: AG object to filter
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Filtered AG object or None if no data in range
        """
        try:
            date_column = "Date"
            print(f"\n🔎 Filtering to date range: {start_date} to {end_date}")

            # Find indices for date range
            start_index = None
            end_index = None

            for i, state in enumerate(agentics.states):
                state_date = getattr(state, date_column, None)
                if state_date:
                    if start_index is None and state_date >= start_date:
                        start_index = i
                    if state_date <= end_date:
                        end_index = i + 1

            if start_index is None or end_index is None:
                print(f"❌ ERROR: No data found for date range {start_date} to {end_date}")
                return None

            num_rows = end_index - start_index
            print(f"📊 Found {num_rows:,} rows in date range")

            # Filter to date range
            filtered_agentics = agentics.filter_states(start=start_index, end=end_index)
            print(f"✅ Filtered to {len(filtered_agentics.states):,} states")

            return filtered_agentics

        except Exception as e:
            print(f"❌ ERROR filtering by date: {e}")
            traceback.print_exc()
            return None

    def _apply_sampling_if_needed(self, agentics: AG, target_size: int = 100) -> AG:
        """
        Apply chunk-based sampling if dataset is too large.

        Args:
            agentics: AG object to sample
            target_size: Target number of rows after sampling

        Returns:
            Sampled AG object
        """
        actual_rows = len(agentics.states)

        if actual_rows <= target_size:
            print(f"✅ No sampling needed ({actual_rows} rows <= {target_size})")
            return agentics

        print(f"\n🎲 Dataset exceeds {target_size} rows. Applying chunk-based sampling...")
        print(f"   Original rows: {actual_rows:,}")

        chunk_size = actual_rows / target_size
        num_chunks = target_size
        print(f"   Chunk size: ~{chunk_size:.1f} rows per chunk")
        print(f"   Will select {num_chunks} rows")

        def count_non_null_values(state: BaseModel) -> int:
            """Count non-null, non-empty values in a state."""
            count = 0
            for field_name, field_value in state.model_dump().items():
                if field_value is not None and field_value != "":
                    count += 1
            return count

        # Sample by chunks
        sampled_states = []
        for chunk_idx in range(num_chunks):
            start_idx = int(chunk_idx * chunk_size)
            end_idx = int((chunk_idx + 1) * chunk_size) if chunk_idx < num_chunks - 1 else actual_rows

            chunk_states = agentics.states[start_idx:end_idx]

            if chunk_states:
                best_state = max(chunk_states, key=count_non_null_values)
                sampled_states.append(best_state)

        # Create new AG with sampled states
        sampled_agentics = agentics.clone()
        sampled_agentics.states = sampled_states

        sampled_rows = len(sampled_agentics.states)
        print(f"✅ Sampled down to {sampled_rows:,} rows")

        return sampled_agentics

    async def answer_question(
        self,
        question: str,
        start_date: str,
        end_date: str,
        selected_columns: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using transduction over the specified date range.

        Args:
            question: The question to answer
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            selected_columns: Optional list of columns to focus on
            conversation_history: Optional list of previous Q&A pairs

        Returns:
            Dict containing answer and metadata
        """
        try:
            print(f"\n{'='*80}")
            print(f"🔍 Transduction Analysis Pipeline")
            print(f"{'='*80}")
            print(f"Question: '{question}'")
            print(f"Date range: {start_date} to {end_date}")

            # Add conversation history context if provided
            if conversation_history and len(conversation_history) > 0:
                print(f"📝 Including {len(conversation_history)} previous conversation turns")
                # Augment question with conversation context
                context_text = "\n\nPrevious conversation:\n"
                for i, turn in enumerate(conversation_history[-3:], 1):  # Last 3 turns
                    context_text += f"Q{i}: {turn['question']}\n"
                    context_text += f"A{i}: {turn['answer'][:200]}...\n\n"  # Truncate long answers
                context_text += f"Current question: {question}"
                augmented_question = context_text
            else:
                augmented_question = question

            # Load data if not already loaded
            if self.agentics is None:
                success = self.load_data(selected_columns)
                if not success:
                    return {
                        "success": False,
                        "error": "Failed to load data"
                    }
            else:
                # If data is loaded but we have new column selection, reload
                if selected_columns is not None:
                    print(f"\n🔄 Reloading data with new column selection...")
                    success = self.load_data(selected_columns)
                    if not success:
                        return {
                            "success": False,
                            "error": "Failed to reload data with selected columns"
                        }

            # Filter by date range
            filtered_agentics = self._filter_by_date_range(self.agentics, start_date, end_date)
            if filtered_agentics is None:
                return {
                    "success": False,
                    "error": f"No data found for date range {start_date} to {end_date}"
                }

            num_rows = len(filtered_agentics.states)

            # Apply sampling if needed
            sampled_agentics = self._apply_sampling_if_needed(filtered_agentics, target_size=100)
            sampled_rows = len(sampled_agentics.states)

            # Calculate batch configuration
            batch_size = max(2, sampled_rows // 10)
            num_batches = (sampled_rows + batch_size - 1) // batch_size
            print(f"📦 Batch configuration: {num_batches} batches of ~{batch_size} rows each")

            # Generate intermediate answer type
            print(f"\n🤖 Generating dynamic Pydantic model...")
            intermediate_answer_ag = AG()
            intermediate_answer_ag = await intermediate_answer_ag.generate_atype(augmented_question)
            pydantic_class = intermediate_answer_ag.atype

            if pydantic_class is None:
                raise ValueError("Failed to generate Pydantic type")

            print(f"✨ Generated Pydantic class: {pydantic_class.__name__}")

            # Perform batch reduction
            print(f"\n⚡ Starting batch reduction...")
            reducer_ag = AG(
                atype=pydantic_class,
                transduction_type="areduce",
                areduce_batch_size=batch_size,
            )

            reduced = await (reducer_ag << sampled_agentics)

            # Add question to reduced result
            reduced = reduced.add_attribute("question", default_value=question)

            # Display intermediate results
            if hasattr(reduced, 'areduce_batches') and reduced.areduce_batches:
                print(f"\n🔽 Processed {len(reduced.areduce_batches)} batches")
                print(f"🔍 DEBUG - Batch results:")
                for i, batch in enumerate(reduced.areduce_batches[:3]):  # Show first 3 batches
                    print(f"   Batch {i+1}: {batch}")

            print(f"\n🔽 Final reduced result:")
            print(f"🔍 DEBUG - Reduced object details:")
            print(f"   Type: {type(reduced)}")
            print(f"   Number of states: {len(reduced.states) if hasattr(reduced, 'states') else 'N/A'}")
            if hasattr(reduced, 'states') and reduced.states:
                print(f"   First state: {reduced.states[0]}")
            reduced.pretty_print()

            # Generate final answer
            print(f"\n📝 Generating final answer...")
            answer_ag = AG(
                atype=TransductionAnswer,
                transduction_type="areduce"
            )
            answer_ag.instructions = f"""
You are analyzing financial data to answer the following question: "{question}"

Based on the intermediate analysis results provided, generate a comprehensive answer that:
1. Directly addresses the user's question with specific evidence from the data
2. Includes relevant data points, trends, and patterns observed
3. Provides context and interpretation of the findings
4. Is detailed and thorough (at least 3-5 paragraphs)

Also provide a clear explanation of your analytical reasoning and how you derived the answer from the data.
"""
            answer = await (answer_ag << reduced)
            print(f"✅ Answer generation complete")

            # Debug: Check what we got back
            print(f"\n🔍 DEBUG - Answer object:")
            print(f"   Type: {type(answer)}")
            print(f"   Has states: {hasattr(answer, 'states')}")
            if hasattr(answer, 'states'):
                print(f"   Number of states: {len(answer.states)}")
                if answer.states:
                    print(f"   First state type: {type(answer.states[0])}")
                    print(f"   First state: {answer.states[0]}")

            # Extract answer
            final_answer = answer[0] if len(answer) > 0 else None

            if final_answer:
                print("\n" + "=" * 80)
                print("🎉 SUCCESS: Analysis complete!")
                print("=" * 80)
                print(f"\n📊 Final Answer Object:")
                print(f"   Type: {type(final_answer)}")
                print(f"   detailed_answer: {getattr(final_answer, 'detailed_answer', 'NOT FOUND')}")
                print(f"   explanation: {getattr(final_answer, 'explanation', 'NOT FOUND')}")
                if hasattr(final_answer, 'model_dump'):
                    print(f"   Full dump: {final_answer.model_dump()}")
                print("=" * 80)

                return {
                    "success": True,
                    "question": question,
                    "date_range": {
                        "start": start_date,
                        "end": end_date,
                        "total_rows": num_rows,
                        "analyzed_rows": sampled_rows,
                        "sampling_applied": sampled_rows < num_rows,
                        "batch_size": batch_size,
                        "num_batches": num_batches
                    },
                    "detailed_answer": final_answer.detailed_answer,
                    "explanation": final_answer.explanation,
                    "transduction_flow": {
                        "initial_states": len(filtered_agentics.states),
                        "sampled_states": sampled_rows,
                        "intermediate_batches": len(getattr(reduced, 'areduce_batches', [])),
                        "final_states": len(reduced.states)
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to generate answer"
                }

        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()

            print("\n" + "=" * 80)
            print("💥 EXCEPTION in Transduction Pipeline")
            print("=" * 80)
            print(f"❌ Error: {error_msg}")
            print(f"❌ Type: {type(e).__name__}")
            print("\nFull Traceback:")
            print(error_trace)
            print("=" * 80)

            return {
                "success": False,
                "error": error_msg,
                "traceback": error_trace
            }


def run_transduction_analysis(
    question: str,
    start_date: str,
    end_date: str,
    selected_columns: Optional[List[str]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run transduction analysis.

    Args:
        question: The question to answer
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        selected_columns: Optional list of columns to focus on
        conversation_history: Optional list of previous Q&A pairs

    Returns:
        Dict containing answer and metadata
    """
    pipeline = TransductionPipeline()
    return asyncio.run(pipeline.answer_question(
        question=question,
        start_date=start_date,
        end_date=end_date,
        selected_columns=selected_columns,
        conversation_history=conversation_history
    ))

