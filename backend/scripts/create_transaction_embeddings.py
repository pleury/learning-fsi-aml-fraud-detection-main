#!/usr/bin/env python3
"""
Transaction Embeddings Generation Script

This script generates embeddings for all transactions in the database using Azure Foundry
and the same text representation function used by the fraud detection service.

The script will:
1. Connect to MongoDB
2. Retrieve all transactions that don't have embeddings or need to be updated
3. Generate text representations using the _create_transaction_text_representation function
4. Create embeddings using Azure Foundry
5. Update the transactions with the generated embeddings

Usage:
    python scripts/create_transaction_embeddings.py [--batch-size BATCH_SIZE] [--force]

Options:
    --batch-size: Number of transactions to process in each batch (default: 10)
    --force: Re-generate embeddings even if they already exist
"""

import os
import sys
import asyncio
import logging
import argparse
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.mongo_db import MongoDBAccess
from azure_foundry.embeddings import get_embedding
from services.fraud_detection import FraudDetectionService

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('transaction_embeddings.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress all Azure SDK logging
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)

class TransactionEmbeddingGenerator:
    """Generate and store embeddings for transaction data"""
    
    def __init__(self, db_client: MongoDBAccess, db_name: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            db_client: MongoDB client instance
            db_name: Database name (defaults to environment or "threatsight360")
        """
        self.db_client = db_client
        self.db_name = db_name or os.getenv("DB_NAME", "threatsight360")
        self.transaction_collection = "transactions"
        
        # Initialize fraud detection service to use its text representation function
        self.fraud_service = FraudDetectionService(db_client, db_name)
        
        logger.info(f"Initialized TransactionEmbeddingGenerator with database: {self.db_name}")
    
    async def generate_embeddings_for_all_transactions(self, batch_size: int = 10, force: bool = False, limit: int = None):
        """
        Generate embeddings for transactions in the database.
        
        Args:
            batch_size: Number of transactions to process in each batch
            force: If True, regenerate embeddings even if they already exist
            limit: Maximum number of transactions to process (None for all)
        """
        logger.info(f"Starting transaction embedding generation (batch_size={batch_size}, force={force})")
        
        # Get the transactions collection
        collection = self.db_client.get_collection(
            db_name=self.db_name,
            collection_name=self.transaction_collection
        )
        
        # Build query based on force parameter
        if force:
            # Process all transactions
            query = {}
            logger.info("Force mode: Processing ALL transactions")
        else:
            # Only process transactions without embeddings or with empty embeddings
            query = {
                "$or": [
                    {"vector_embedding": {"$exists": False}},
                    {"vector_embedding": None},
                    {"vector_embedding": []},
                    {"vector_embedding": {"$size": 0}}
                ]
            }
            logger.info("Normal mode: Processing only transactions without embeddings")
        
        # Count total transactions to process
        total_count = collection.count_documents(query)
        logger.info(f"Found {total_count} transactions to process")
        
        if total_count == 0:
            logger.info("No transactions need embedding generation")
            return
        
        # Process transactions in batches
        processed_count = 0
        error_count = 0
        batch_num = 1
        
        # Use cursor to efficiently process large datasets
        cursor = collection.find(query).batch_size(batch_size)
        
        batch_transactions = []
        transactions_processed = 0  # Add this counter

        for transaction in cursor:
            # Check limit before adding to batch
            if limit and transactions_processed >= limit:
                break
            
            batch_transactions.append(transaction)
            transactions_processed += 1
            
            # Process batch when it's full or we've reached the limit
            if len(batch_transactions) >= batch_size or (limit and transactions_processed >= limit):
                success, errors = await self._process_batch(batch_transactions, batch_num)
                processed_count += success
                error_count += errors
                batch_num += 1
                batch_transactions = []
                
                # Break if we've hit the limit
                if limit and transactions_processed >= limit:
                    break
                
                # Log progress
                progress = (processed_count + error_count) / min(total_count, limit or total_count) * 100
                logger.info(f"Progress: {processed_count + error_count}/{min(total_count, limit or total_count)} ({progress:.1f}%) - "
                          f"Successful: {processed_count}, Errors: {error_count}")
        
        # Process remaining transactions
        if batch_transactions:
            success, errors = await self._process_batch(batch_transactions, batch_num)
            processed_count += success
            error_count += errors
        
        # Final summary
        logger.info(f"Embedding generation complete!")
        logger.info(f"Total processed: {processed_count + error_count}")
        logger.info(f"Successful: {processed_count}")
        logger.info(f"Errors: {error_count}")
        
        if error_count > 0:
            logger.warning(f"There were {error_count} errors. Check the log file for details.")
    
    async def _process_batch(self, transactions: List[Dict[str, Any]], batch_num: int) -> tuple[int, int]:
        """
        Process a batch of transactions to generate embeddings.
        
        Args:
            transactions: List of transaction documents
            batch_num: Batch number for logging
            
        Returns:
            Tuple of (successful_count, error_count)
        """
        logger.info(f"Processing batch {batch_num} with {len(transactions)} transactions")
        
        successful_count = 0
        error_count = 0
        
        for transaction in transactions:
            try:
                # Generate text representation using the same method as fraud detection
                transaction_text = self.fraud_service._create_transaction_text_representation(transaction)
                
                # Generate embedding using Azure Foundry
                logger.debug(f"Generating embedding for transaction {transaction.get('transaction_id', 'unknown')}")
                logger.debug(f"Transaction text: {transaction_text}")
                embedding = await get_embedding(transaction_text)
                
                # Validate embedding
                if not embedding or not isinstance(embedding, list) or len(embedding) == 0:
                    logger.error(f"Invalid embedding generated for transaction {transaction.get('_id')}")
                    error_count += 1
                    continue
                
                # Update the transaction with the embedding
                result = self.db_client.get_collection(
                    db_name=self.db_name,
                    collection_name=self.transaction_collection
                ).update_one(
                    {"_id": transaction["_id"]},
                    {
                        "$set": {
                            "vector_embedding": embedding,
                            "embedding_metadata": {
                                "generated_at": datetime.now(),
                                "embedding_model": "azure-foundry",
                                "embedding_dimension": len(embedding),
                                "text_representation_method": "_create_transaction_text_representation"
                            }
                        }
                    }
                )
                
                if result.modified_count == 1:
                    successful_count += 1
                    logger.debug(f"Successfully updated transaction {transaction.get('transaction_id', 'unknown')} "
                              f"with {len(embedding)}-dimensional embedding")
                else:
                    logger.error(f"Failed to update transaction {transaction.get('_id')} in database")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing transaction {transaction.get('_id', 'unknown')}: {str(e)}")
                error_count += 1
                continue
        
        logger.info(f"Batch {batch_num} complete: {successful_count} successful, {error_count} errors")
        return successful_count, error_count
    
    async def verify_embeddings(self):
        """
        Verify that embeddings were generated correctly.
        """
        logger.info("Verifying embeddings...")
        
        collection = self.db_client.get_collection(
            db_name=self.db_name,
            collection_name=self.transaction_collection
        )
        
        # Count total transactions
        total_transactions = collection.count_documents({})
        
        # Count transactions with embeddings
        transactions_with_embeddings = collection.count_documents({
            "vector_embedding": {"$exists": True, "$ne": None, "$not": {"$size": 0}}
        })
        
        # Get sample embedding dimensions
        sample_transaction = collection.find_one({
            "vector_embedding": {"$exists": True, "$ne": None, "$not": {"$size": 0}}
        })
        
        sample_dimension = len(sample_transaction["vector_embedding"]) if sample_transaction else 0
        
        logger.info(f"Verification Results:")
        logger.info(f"  Total transactions: {total_transactions}")
        logger.info(f"  Transactions with embeddings: {transactions_with_embeddings}")
        logger.info(f"  Coverage: {transactions_with_embeddings/total_transactions*100:.1f}%")
        logger.info(f"  Sample embedding dimension: {sample_dimension}")
        
        if sample_dimension > 0:
            logger.info("✅ Embeddings verification successful!")
        else:
            logger.warning("⚠️ No valid embeddings found!")
        
        return transactions_with_embeddings, total_transactions


async def main():
    """Main function to run the embedding generation script"""
    parser = argparse.ArgumentParser(description="Generate embeddings for transactions")
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=10, 
        help="Number of transactions to process in each batch (default: 10)"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Re-generate embeddings even if they already exist"
    )
    parser.add_argument(
        "--verify-only", 
        action="store_true", 
        help="Only verify existing embeddings without generating new ones"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        help="Maximum number of transactions to process (useful for testing)"
    )
    
    args = parser.parse_args()
    
    # Check for required environment variables
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        logger.error("MONGODB_URI environment variable is not set")
        sys.exit(1)
    
    # Check Azure Foundry configuration
    azure_endpoint = os.getenv("INFERENCE_ENDPOINT") 
    azure_api_key = os.getenv("AZURE_AI_API_KEY")
    if not azure_endpoint or not azure_api_key:
        logger.error("Azure Foundry environment variables (INFERENCE_ENDPOINT, AZURE_AI_API_KEY) are not set")
        sys.exit(1)
    
    logger.info("Starting Transaction Embedding Generation Script")
    logger.info(f"MongoDB URI: {mongodb_uri[:20]}...")
    logger.info(f"Azure Endpoint: {azure_endpoint}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Force regeneration: {args.force}")
    
    try:
        # Initialize database connection
        db_client = MongoDBAccess(mongodb_uri)
        
        # Initialize embedding generator
        generator = TransactionEmbeddingGenerator(db_client)
        
        if args.verify_only:
            # Only verify existing embeddings
            await generator.verify_embeddings()
        else:
            # Generate embeddings
            await generator.generate_embeddings_for_all_transactions(
                batch_size=args.batch_size,
                force=args.force,
                limit=args.limit
            )
            
            # Verify the results
            await generator.verify_embeddings()
        
        logger.info("Script completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
