import asyncio
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

async def process_images_batch_async(
    image_mappings: Dict[str, Dict[str, Any]],  # Changed from List to Dict
    llm: Any,  # Your LLM client object
    model_name: str,
    function_schema: dict,
    extraction_system_prompt: str,
    extraction_human_prompt: str,
    max_concurrent: int = 3
) -> List[Dict[str, Any]]:
    """
    Process multiple images with concurrency control.
    
    Args:
        image_mappings: Dictionary of image mappings keyed by image_id
        llm: OpenAI async client
        model_name: Vision-capable model name
        function_schema: The extraction function schema
        extraction_system_prompt: System prompt from main extraction
        extraction_human_prompt: Human prompt template from main extraction
        max_concurrent: Maximum concurrent image processing tasks
        
    Returns:
        List of extraction results
    """
    results = []
    
    # Process in batches to avoid rate limits
    # Convert dictionary items to list for batch processing
    image_items = list(image_mappings.items())
    
    for i in range(0, len(image_items), max_concurrent):
        batch_items = image_items[i:i + max_concurrent]
        batch_tasks = []
        
        for image_id, image_data in batch_items:
            task = process_image_async(
                image_id=image_id,
                image_data=image_data,
                llm=llm,
                model_name=model_name,
                function_schema=function_schema,
                extraction_system_prompt=extraction_system_prompt,
                extraction_human_prompt=extraction_human_prompt
            )
            batch_tasks.append(task)
        
        # Wait for batch to complete
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
    
    # Small delay between batches to avoid rate limits
    if i + max_concurrent < len(image_items):
        await asyncio.sleep(1)
    
    return results


async def process_image_async(
    image_id: str,
    image_data: Dict[str, Any],
    llm: Any,
    model_name: str,
    function_schema: dict,
    extraction_system_prompt: str,
    extraction_human_prompt: str
) -> Dict[str, Any]:
    """
    Process a single image asynchronously.
    
    Args:
        image_id: The ID of the image
        image_data: Dictionary containing image content and metadata
        llm: OpenAI async client
        model_name: Vision-capable model name
        function_schema: The extraction function schema
        extraction_system_prompt: System prompt from main extraction
        extraction_human_prompt: Human prompt template from main extraction
        
    Returns:
        Dictionary with extraction results
    """
    try:
        # Extract base64 data and metadata
        base64_data = image_data.get("content", "")
        metadata = image_data.get("metadata", {})
        
        # Format the image for the API call
        image_format = metadata.get("format", "png")
        image_url = f"data:image/{image_format};base64,{base64_data}"
        
        # Create the messages for the LLM
        messages = [
            {"role": "system", "content": extraction_system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": extraction_human_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
        
        # Make the API call
        response = await llm.chat.completions.create(
            model=model_name,
            messages=messages,
            functions=[function_schema],
            function_call={"name": function_schema["name"]}
        )
        
        # Extract the function call results
        function_call = response.choices[0].message.function_call
        if function_call:
            import json
            extracted_data = json.loads(function_call.arguments)
        else:
            extracted_data = {}
        
        return {
            "image_id": image_id,
            "extraction_results": extracted_data,
            "metadata": metadata,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error processing image {image_id}: {str(e)}")
        return {
            "image_id": image_id,
            "extraction_results": {},
            "metadata": metadata,
            "status": "error",
            "error_message": str(e)
        }


# Alternative version if you want to maintain the original function signature
async def process_images_batch_async_legacy(
    images: List[Any],  # List of ImageReference objects
    llm: Any,
    model_name: str,
    function_schema: dict,
    extraction_system_prompt: str,
    extraction_human_prompt: str,
    max_concurrent: int = 3
) -> List[Dict[str, Any]]:
    """
    Legacy version that accepts a list of ImageReference objects.
    Converts them to dictionary format and processes.
    """
    # Convert ImageReference list to dictionary format
    image_mappings = {}
    for img in images:
        image_mappings[img.img_id] = {
            "content": img.base64_data,
            "metadata": {
                "format": img.img_format,
                "size_bytes": img.size_bytes,
                "estimated_tokens": img.estimated_tokens,
                "placeholder": img.placeholder
            }
        }
    
    # Process using the dictionary-based function
    return await process_images_batch_async(
        image_mappings=image_mappings,
        llm=llm,
        model_name=model_name,
        function_schema=function_schema,
        extraction_system_prompt=extraction_system_prompt,
        extraction_human_prompt=extraction_human_prompt,
        max_concurrent=max_concurrent
    )


# Helper function to convert results back to a dictionary keyed by image_id
def results_to_dict(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Convert list of results to dictionary keyed by source_id (image_id).
    
    Args:
        results: List of extraction results
        
    Returns:
        Dictionary keyed by source_id
    """
    results_dict = {}
    for result in results:
        source_id = result.get("source_id")
        if source_id:
            results_dict[source_id] = result
    return results_dict


# Example usage
async def main():
    # Assuming you have image_mappings from extract_and_map_sources
    chunk_mappings, image_mappings = extract_and_map_sources(html_content)
    
    # Process images
    results = await process_images_batch_async(
        image_mappings=image_mappings,
        llm=your_llm_client,
        model_name="gpt-4-vision-preview",
        function_schema=your_function_schema,
        extraction_system_prompt="Extract company mentions from images",
        extraction_human_prompt="Extract all company names from this image",
        max_concurrent=3
    )
    
    # Results will be a list of flat dictionaries like:
    # [
    #     {
    #         "source_id": "img_001",
    #         "source_type": "image",
    #         "companies": [{"Word": "Nvidia", "BIC_value": "NVDA.O", ...}],
    #         "BSTicker_value": "MSFT US",
    #         "BSTicker_confidence": "very_high",
    #         # ... all other extracted fields
    #     },
    #     ...
    # ]
    
    # Convert results to dictionary if needed
    results_dict = results_to_dict(results)
    
    # Access specific result
    if "img_001" in results_dict:
        print(results_dict["img_001"])  # All fields are at top level now
