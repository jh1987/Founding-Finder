import numpy as np
import pandas as pd
import hnswlib
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import torch
import sys
import gc
import warnings
from huggingface_hub import hf_hub_download, HfApi, HfFolder, Repository, hf_hub_url

# Suppress warnings
warnings.filterwarnings('ignore')

# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure PyTorch
torch.set_grad_enabled(False)  # Disable gradients globally
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear CUDA cache
    
# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the RAG engine with a sentence transformer model."""
        try:
            # Set device before model initialization
            self.device = 'cpu'  # Force CPU usage to avoid CUDA initialization issues
            
            # Initialize model with device specification
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # Disable gradient computation for all parameters
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Initialize other attributes
            self.index = None
            self.funding_data = None
            self.embeddings = None
            
            # Initialize OpenAI client
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
        except Exception as e:
            print(f"Error initializing RAG engine: {str(e)}", file=sys.stderr)
            raise RuntimeError(f"Failed to initialize RAG engine: {str(e)}")
        
    def __del__(self):
        """Cleanup resources when the object is destroyed."""
        try:
            # Clear CUDA cache if it was used
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear model from memory
            self.model = None
            self.index = None
            self.embeddings = None
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}", file=sys.stderr)

    def prepare_funding_text(self, row: pd.Series) -> str:
        """Convert a funding opportunity row into a text representation."""
        return f"""
        Program: {row['Funding Program Name']}
        Type: {row['Funding Type']}
        Amount: {row['Funding Amount/Range']}
        Eligibility: {row['Eligibility Criteria']}
        Industry: {row['Industry Focus']}
        Location: {row['Geographical Restrictions']}
        Benefits: {row['Additional Benefits']}
        """

    def load_and_embed_data(self, csv_path: str) -> None:
        """Load funding data from CSV and generate embeddings."""
        try:
            # Clear any existing data
            if self.embeddings is not None:
                del self.embeddings
                self.embeddings = None
                gc.collect()
            
            # Load the CSV data
            self.funding_data = pd.read_csv(csv_path)
            
            # Generate text representations
            texts = [self.prepare_funding_text(row) for _, row in self.funding_data.iterrows()]
            
            # Generate embeddings with progress bar
            print("Generating embeddings...")
            self.embeddings = []
            
            # Process in smaller batches to avoid memory issues
            batch_size = 8  # Reduced batch size
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i + batch_size]
                with torch.no_grad():  # Ensure no gradients are computed
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                self.embeddings.extend(batch_embeddings)
                
                # Clear memory after each batch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            
            # Convert to numpy array
            self.embeddings = np.array(self.embeddings).astype('float32')
            
            # Create HNSW index
            if self.index is not None:
                del self.index
                gc.collect()
            
            # Initialize the HNSW index
            self.index = hnswlib.Index(space='l2', dim=self.model.get_sentence_embedding_dimension())
            self.index.init_index(max_elements=len(self.embeddings), ef_construction=200, M=16)
            self.index.add_items(self.embeddings)
            
            print(f"Loaded {len(self.funding_data)} funding opportunities and created embeddings.")
            
        except Exception as e:
            print(f"Error loading and embedding data: {str(e)}", file=sys.stderr)
            raise RuntimeError(f"Failed to load and embed data: {str(e)}")

    def check_eligibility(self, opportunity: Dict, quiz_responses: Dict) -> Tuple[bool, str]:
        """Use AI to determine if a startup is eligible for a funding opportunity."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in startup funding eligibility analysis. Consider the criteria carefully but be somewhat flexible in your assessment, giving startups the benefit of the doubt when criteria are not explicitly exclusionary."},
                    {"role": "user", "content": f"""
                        Analyze if the startup is eligible for this funding opportunity. Consider all criteria carefully but be somewhat flexible.

                        Startup Profile:
                        - Name: {quiz_responses['startup_name']}
                        - Industry: {quiz_responses['industry']}
                        - Stage: {quiz_responses['stage']}
                        - Location: {quiz_responses['location']}
                        - Funding Needed: {quiz_responses['funding_needed']} EUR

                        Funding Opportunity:
                        - Program: {opportunity['Funding Program Name']}
                        - Type: {opportunity['Funding Type']}
                        - Amount Range: {opportunity['Funding Amount/Range']}
                        - Eligibility Criteria: {opportunity['Eligibility Criteria']}
                        - Industry Focus: {opportunity['Industry Focus']}
                        - Geographical Restrictions: {opportunity['Geographical Restrictions']}

                        First, determine if the startup is eligible based on:
                        1. Location requirements (if not explicitly restricted, consider it eligible)
                        2. Industry focus (if broad or related industries, consider it eligible)
                        3. Stage/maturity requirements (if not specified, consider it eligible)
                        4. Funding amount limits (if within ±30% of range, consider it eligible)
                        5. Any other specific eligibility criteria

                        Return your response in the following format:
                        ELIGIBLE: true/false
                        REASON: Brief explanation of eligibility or why they're not eligible
                    """}
                ],
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            eligible_line = [line for line in result.split('\n') if line.startswith('ELIGIBLE:')][0]
            reason_line = [line for line in result.split('\n') if line.startswith('REASON:')][0]
            
            is_eligible = 'true' in eligible_line.lower()
            reason = reason_line.replace('REASON:', '').strip()
            
            return is_eligible, reason
            
        except Exception as e:
            print(f"Error in eligibility check: {str(e)}", file=sys.stderr)
            return True, "Eligibility check failed, including by default"

    def prepare_query_text(self, quiz_responses: Dict) -> str:
        """Convert quiz responses into a search query text."""
        return f"""
        Looking for funding with the following criteria:
        Industry: {quiz_responses.get('industry', '')}
        Stage: {quiz_responses.get('stage', '')}
        Funding Type Needed: {quiz_responses.get('funding_type', '')}
        Amount Needed: {quiz_responses.get('funding_needed', '')} EUR
        Location: {quiz_responses.get('location', '')}
        """

    def search(self, quiz_responses: Dict, top_k: int = 10) -> List[Dict]:
        """Search for matching funding opportunities based on quiz responses."""
        try:
            if self.index is None:
                raise ValueError("No data loaded. Please load data first.")
            
            # Convert quiz responses to query text and generate embedding
            query_text = self.prepare_query_text(quiz_responses)
            print(f"Searching with query: {query_text}")
            
            with torch.no_grad():
                query_embedding = self.model.encode(
                    [query_text],
                    convert_to_numpy=True,
                    show_progress_bar=False
                )[0].astype('float32')
            
            # Perform similarity search with increased k to ensure we have enough candidates
            k = min(top_k * 3, len(self.funding_data))  # Triple the search space but don't exceed dataset size
            indices, distances = self.index.knn_query(query_embedding, k=k)
            indices = indices[0]
            distances = distances[0]
            
            print(f"Found {len(indices)} initial matches")
            
            # Get matching opportunities and check eligibility
            matches = []
            for idx, distance in zip(indices, distances):
                match = self.funding_data.iloc[idx].to_dict()
                print(f"\nChecking eligibility for: {match['Funding Program Name']}")
                
                # Check eligibility using AI
                is_eligible, reason = self.check_eligibility(match, quiz_responses)
                print(f"Eligible: {is_eligible}, Reason: {reason}")
                
                if is_eligible:
                    match['similarity_score'] = 1 / (1 + distance)
                    match['eligibility_reason'] = reason
                    matches.append(match)
                    print(f"Added to matches. Current matches: {len(matches)}")
                
                # Return top 5 eligible matches
                if len(matches) >= 5:
                    break
            
            # If no matches found, return the top matches without eligibility check
            if not matches and len(indices) > 0:
                print("No eligible matches found. Returning top matches without strict eligibility check.")
                for idx, distance in zip(indices[:5], distances[:5]):
                    match = self.funding_data.iloc[idx].to_dict()
                    match['similarity_score'] = 1 / (1 + distance)
                    match['eligibility_reason'] = "Potential match, please review eligibility criteria carefully"
                    matches.append(match)
            
            # Clear some memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            print(f"Returning {len(matches)} matches")
            return matches
            
        except Exception as e:
            print(f"Error in search: {str(e)}", file=sys.stderr)
            raise RuntimeError(f"Failed to perform search: {str(e)}")

    def generate_recommendation(self, quiz_responses: Dict, matches: List[Dict]) -> str:
        """Generate a personalized recommendation using OpenAI's API."""
        try:
            if not matches:
                return "No eligible funding opportunities found matching your criteria."
                
            # Prepare context for the prompt
            context = f"""
            User Profile:
            - Industry: {quiz_responses['industry']}
            - Stage: {quiz_responses['stage']}
            - Funding Type Needed: {quiz_responses['funding_type']}
            - Amount Needed: {quiz_responses['funding_needed']} EUR
            - Location: {quiz_responses['location']}

            Top Matching Funding Opportunities:
            """
            
            for match in matches:
                context += f"""
                Program: {match['Funding Program Name']}
                Type: {match['Funding Type']}
                Amount: {match['Funding Amount/Range']}
                Eligibility: {match['Eligibility Criteria']}
                Industry Focus: {match['Industry Focus']}
                Location: {match['Geographical Restrictions']}
                Benefits: {match['Additional Benefits']}
                Eligibility Analysis: {match['eligibility_reason']}
                """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a startup funding expert helping match startups with appropriate funding opportunities. Be specific about eligibility criteria and requirements."},
                    {"role": "user", "content": f"""
                        Based on the startup's profile and the matching funding opportunities provided, generate a personalized 
                        recommendation explaining why each funding option is suitable. Focus on:
                        1. How well each option matches their needs
                        2. Key eligibility criteria they meet
                        3. Specific benefits that align with their situation
                        4. Any important next steps or considerations
                        5. The eligibility analysis for each option

                        {context}

                        Please provide a concise but informative recommendation that helps the startup understand 
                        why these options are good matches and what they should consider next. Include specific 
                        eligibility requirements they meet for each option.
                    """}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating recommendation: {str(e)}", file=sys.stderr)
            return f"Error generating recommendation: {str(e)}"