from crewai.tools import BaseTool
from typing import Type, Optional, List
from pydantic import BaseModel, Field
from agentics import AG
import pandas as pd
import asyncio
import os
import csv
import sys
import traceback

# Module-level variable to store selected columns from UI
# This is set by the Streamlit app before running analysis
class TransductionAnswer(BaseModel):
    detailed_answer: str | None = Field(
        None,
        description="A detailed multi-paragraph answer with thorough evidence from the dataset."
    )
    explanation: str | None = Field(
        None,
        description="A detailed reasoning and explanation of the above answer."
    )


class TransductionPipeline:
    def run(self, question: str) -> dict:
        try:
            # Get selected columns and date range from UI (set deterministically by user selection)
            from pipeline.runner import get_selected_columns, get_date_range
            selected_columns = get_selected_columns()
            date_range = get_date_range()

            # Use UI-provided date range if available, otherwise use defaults
            if date_range:
                start_date, end_date = date_range
            else:
                start_date = "2018-01-01"
                end_date = "2025-01-01"

            print(f"\n{'='*80}")
            print(f"🔍 Starting Transduction Analysis")
            print(f"{'='*80}")
            print(f"Question: '{question}'")
            print(f"Date range: {start_date} to {end_date}")

            # Use the merged dataset (supports both single file and split files)
            date_column = "Date"
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

            # Import the CSV reader utility
            from utils.csv_reader import get_merged_data_file_for_agentics

            try:
                csv_path = get_merged_data_file_for_agentics(data_dir)
            except FileNotFoundError as e:
                print(f"❌ ERROR: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

            # Increase CSV field size limit to handle large text fields (e.g., Headlines, Earningcall)
            # Default limit is 131072 bytes (128KB), increase to handle large text content
            # Set to a large value (10MB) to accommodate very long text fields
            try:
                csv.field_size_limit(10 * 1024 * 1024)  # 10MB
            except OverflowError:
                # If 10MB is too large for the system, use maximum allowed
                csv.field_size_limit(sys.maxsize)

            # Load the full dataset first
            print(f"\n📂 Loading merged dataset")
            print(f"📅 Date column: {date_column}")
            agentics = AG.from_csv(csv_path)
            print(f"✅ Loaded {len(agentics.states):,} total rows with {len(agentics.atype.model_fields)} columns")

            # Filter to selected columns if provided (using Agentics __call__ method)
            if selected_columns is not None and len(selected_columns) > 0:
                # Agentics sanitizes column names when creating Pydantic models
                # (removes special characters like ^, -, etc.)
                # We need to sanitize the column names to match what's in the model
                import re
                def sanitize_field_name(name: str) -> str:
                    """Sanitize field name to match Agentics' sanitization."""
                    name = name.strip()
                    # Remove underscores only from the start
                    name = re.sub(r"^_+", "", name)
                    # If the result is alphanumeric, return as-is
                    if re.fullmatch(r"[a-zA-Z0-9_]+", name):
                        return name
                    # Otherwise, remove all non-alphanumeric and non-underscore characters
                    return re.sub(r"[^\w]", "", name)

                # Sanitize all selected columns to match Agentics model field names
                sanitized_selected = [sanitize_field_name(col) for col in selected_columns]

                # Always include Date column for filtering, and ensure it comes first
                # Remove Date from sanitized_selected if present to avoid duplicates
                selected_without_date = [col for col in sanitized_selected if col != "Date"]
                # Put Date first, then the rest of the selected columns
                columns_to_include = ["Date"] + selected_without_date
                print(f"\n🔧 Filtering to {len(columns_to_include)} selected columns: {', '.join(columns_to_include[:5])}...")
                print(f"   Original columns: {selected_columns[:5]}...")
                print(f"   Sanitized columns: {columns_to_include[:5]}...")

                # Use Agentics __call__ method to filter to only selected columns
                # This creates a new AG with only the specified fields
                # The order of arguments determines the order of fields in the model
                agentics = agentics(*columns_to_include)
                print(f"✅ Filtered dataset to {len(columns_to_include)} columns (Date first)")
            else:
                print(f"❌ You must select at least one column to analyze")
                return {
                    "success": False,
                    "error": "You must select at least one column to analyze"
                }

            # Find the indices that correspond to start_date and end_date
            # Agentics creates states where each state has attributes corresponding to CSV columns
            # We need to find which states have Date values in our range
            print(f"\n🔎 Searching for date range indices using column '{date_column}'...")
            start_index = None
            end_index = None

            for i, state in enumerate(agentics.states):
                state_date = getattr(state, date_column, None)
                if state_date:
                    # Compare dates as strings (they're in YYYY-MM-DD format)
                    if start_index is None and state_date >= start_date:
                        start_index = i
                    if state_date <= end_date:
                        end_index = i + 1  # +1 because filter_states uses [start:end)

            if start_index is None or end_index is None:
                print(f"❌ ERROR: No data found for date range {start_date} to {end_date}")
                return {
                    "success": False,
                    "error": f"No data found for date range {start_date} to {end_date}"
                }

            # Calculate the number of rows in our date range
            num_rows = end_index - start_index
            print(f"📊 Found date range: rows {start_index} to {end_index} ({num_rows:,} total rows)")

            # Filter the dataset to the date range first
            print(f"\n🔽 Filtering dataset to date range...")
            filtered_agentics = agentics.filter_states(start=start_index, end=end_index)
            print(f"✅ Filtered dataset contains {len(filtered_agentics.states):,} states")

            # Apply chunk-based sampling if we have too many rows
            # For each chunk, pick the row with the most non-null cells
            TARGET_SAMPLE_SIZE = 100
            actual_rows = len(filtered_agentics.states)

            if actual_rows > TARGET_SAMPLE_SIZE:
                print(f"\n🎲 Dataset exceeds {TARGET_SAMPLE_SIZE} rows. Applying chunk-based sampling...")
                print(f"   Original rows: {actual_rows:,}")

                # Calculate chunk size
                chunk_size = actual_rows / TARGET_SAMPLE_SIZE
                num_chunks = TARGET_SAMPLE_SIZE
                print(f"   Chunk size: ~{chunk_size:.1f} rows per chunk")
                print(f"   Will select {num_chunks} rows (one per chunk with most non-null values)")

                # Helper function to count non-null values in a state
                def count_non_null_values(state: BaseModel) -> int:
                    """Count the number of non-null, non-empty values in a Pydantic model."""
                    count = 0
                    for field_name, field_value in state.model_dump().items():
                        if field_value is not None and field_value != "":
                            count += 1
                    return count

                # Sample by chunks: for each chunk, pick the row with most non-null values
                sampled_states = []
                for chunk_idx in range(num_chunks):
                    start_idx = int(chunk_idx * chunk_size)
                    end_idx = int((chunk_idx + 1) * chunk_size) if chunk_idx < num_chunks - 1 else actual_rows

                    # Get all states in this chunk
                    chunk_states = filtered_agentics.states[start_idx:end_idx]

                    if chunk_states:
                        # Find the state with the most non-null values
                        best_state = max(chunk_states, key=count_non_null_values)
                        sampled_states.append(best_state)

                # Create new AG object with sampled states
                filtered_agentics = filtered_agentics.clone()
                filtered_agentics.states = sampled_states

                sampled_rows = len(filtered_agentics.states)
                print(f"✅ Sampled down to {sampled_rows:,} rows (chunk-based, selecting rows with most data)")
            else:
                sampled_rows = actual_rows
                print(f"✅ No sampling needed ({actual_rows} rows <= {TARGET_SAMPLE_SIZE})")

            # Calculate batch size: aim for ~10 batches
            batch_size = max(2, sampled_rows // 10)  # At least 2 rows per batch
            num_batches = (sampled_rows + batch_size - 1) // batch_size  # Ceiling division
            print(f"📦 Batch configuration: {num_batches} batches of ~{batch_size} rows each")

            # Generate intermediate answer type based on the question
            print(f"\n🤖 Generating dynamic Pydantic model for intermediate analysis...")
            try:
                intermediate_answer_ag = AG()
                intermediate_answer_ag = asyncio.run(
                    intermediate_answer_ag.generate_atype(question)
                )
                pydantic_class = intermediate_answer_ag.atype
                pydantic_code = intermediate_answer_ag.atype_code

                if pydantic_class is None:
                    raise ValueError("generate_atype returned None - LLM failed to generate a valid Pydantic type")

                print(f"\n✨ Generated Pydantic class: {pydantic_class.__name__}")
            except Exception as e:
                error_msg = str(e)
                error_trace = traceback.format_exc()

                print("\n" + "=" * 80)
                print("💥 EXCEPTION OCCURRED during Pydantic Model Generation")
                print("=" * 80)
                print(f"❌ Error: {error_msg}")
                print("\nFull Traceback:")
                print(error_trace)
                print("=" * 80 + "\n")

                # Re-raise to be caught by outer exception handler
                raise
            print("=" * 80)
            print("DYNAMIC PYDANTIC MODEL CODE:")
            print("=" * 80)
            print(f"{pydantic_code}")
            print("=" * 80)

            # Log the model fields
            if hasattr(pydantic_class, 'model_fields'):
                print(f"📋 Model fields: {list(pydantic_class.model_fields.keys())}")

            # Perform reduction on the filtered (and possibly sampled) dataset
            print(f"\n⚡ Starting batch reduction with {num_batches} batches...")

            # Create AG with areduce_batch_size
            reducer_ag = AG(
                atype=pydantic_class,
                transduction_type="areduce",
                areduce_batch_size=batch_size,
            )

            reduced = asyncio.run(reducer_ag << filtered_agentics)

            # Store transduction flow data for visualization
            # This will be accessed by the Streamlit app to display the flow
            try:
                import streamlit as st
                # Store flow data in session state
                flow_data = {
                    "initial_states": {
                        "agentics": filtered_agentics,
                        "atype_name": filtered_agentics.atype.__name__ if filtered_agentics.atype else "Unknown",
                        "num_rows": len(filtered_agentics.states)
                    },
                    "intermediate_batches": {
                        "batches": getattr(reduced, 'areduce_batches', []),
                        "atype_name": pydantic_class.__name__ if pydantic_class else "Unknown",
                        "num_batches": len(getattr(reduced, 'areduce_batches', []))
                    },
                    "final_intermediate": {
                        "agentics": reduced,
                        "atype_name": reduced.atype.__name__ if reduced.atype else "Unknown",
                        "num_rows": len(reduced.states)
                    }
                }
                st.session_state.transduction_flow = flow_data
            except Exception as e:
                # If streamlit is not available (e.g., during testing), just continue
                pass

            # Add question attribute to the reduced result
            reduced = reduced.add_attribute("question", default_value=question)

            # Display intermediate batch results if available
            # Note: areduce_batches contains the intermediate results from each batch
            # These are the results after reducing each batch of batch_size rows
            if hasattr(reduced, 'areduce_batches') and reduced.areduce_batches:
                print(f"\n 🔽 INTERMEDIATE BATCH RESULTS ({len(reduced.areduce_batches)} batches):")
                for i, batch_result in enumerate(reduced.areduce_batches, 1):
                    print(f"\n--- Batch {i} ---")
                    if hasattr(batch_result, 'model_dump_json'):
                        print(batch_result.model_dump_json(indent=2))
                    else:
                        print(str(batch_result))

            print(f"\n 🔽 FINAL REDUCED RESULT:")
            reduced.pretty_print()

            # Generate final answer
            print(f"\n📝 Generating final comprehensive answer...")
            answer = asyncio.run(
                AG(atype=TransductionAnswer, transduction_type="areduce") << reduced
            )
            print(f"✅ Final answer generation complete")

            # Store final answer in flow data
            try:
                import streamlit as st
                if st.session_state.transduction_flow:
                    st.session_state.transduction_flow["final_answer"] = {
                        "agentics": answer,
                        "atype_name": answer.atype.__name__ if answer.atype else "Unknown",
                        "num_rows": len(answer.states)
                    }
            except Exception:
                pass

            # Extract the answer from the AG object
            final_answer = answer[0] if len(answer) > 0 else None

            if final_answer:
                print("\n" + "=" * 80)
                print("🎉 SUCCESS: Analysis complete!")
                print("=" * 80)

                result = {
                    "success": True,
                    "date_range": {
                        "start": start_date,
                        "end": end_date,
                        "total_rows_in_range": num_rows,
                        "rows_analyzed": sampled_rows,
                        "sampling_applied": sampled_rows < num_rows,
                        "sampling_ratio": f"1:{int(num_rows/sampled_rows)}" if sampled_rows < num_rows else "1:1",
                        "batch_size": batch_size,
                        "num_batches": num_batches
                    },
                    "question": question,
                    "detailed_answer": final_answer.detailed_answer,
                    "explanation": final_answer.explanation
                }

                print(f"✅ Returning result with {len(result)} fields\n")
                print(result)
                return result
            else:
                print("❌ ERROR: Failed to generate answer - empty result from transduction\n")
                return {
                    "success": False,
                    "error": "Failed to generate answer"
                }

        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()

            print("\n" + "=" * 80)
            print("💥 EXCEPTION OCCURRED during Transduction Analysis")
            print("=" * 80)
            print(f"❌ Error: {error_msg}")
            print(f"❌ Error Type: {type(e).__name__}")
            print("\nFull Traceback:")
            print(error_trace)
            print("=" * 80 + "\n")

            return {
                "success": False,
                "error": error_msg,
                "traceback": error_trace
            }