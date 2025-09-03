from bs4 import BeautifulSoup
import base64
import logging
import tiktoken

logger = logging.getLogger(__name__)

def extract_and_map_sources(html_content: str, max_chunk_tokens: int = 3000):
    """
    Extract chunks and images from HTML content and create source mappings.
    
    Args:
        html_content: HTML content with text and base64 images
        max_chunk_tokens: Maximum tokens per text chunk
        
    Returns:
        Tuple of (chunk_mappings, image_mappings)
    """
    chunk_mappings = []
    image_mappings = []
    
    # Step 1: Extract images and get cleaned text
    cleaned_text, images = extract_images_from_html(html_content)
    logger.info(f"Extracted {len(images)} images from HTML content")
    
    # Step 2: Create text chunk mappings
    if cleaned_text and cleaned_text.strip():
        cleaned_text = ensure_text_is_string(cleaned_text)
        chunks = chunk_text_by_tokens(
            text=cleaned_text,
            max_chunk_tokens=max_chunk_tokens,
            model_encoding="cl100k_base"
        )
        logger.info(f"Split text into {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i:03d}"
            chunk_mapping = {
                "chunk_id": chunk_id,
                "chunk_content": chunk,
                "source_metadata": {
                    "type": "text",
                    "index": i,
                    "length": len(chunk)
                }
            }
            chunk_mappings.append(chunk_mapping)
    
    # Step 3: Create image mappings - modified to match ImageReference structure
    for i, image_ref in enumerate(images):
        image_mapping = {
            "image_id": image_ref.img_id,  # Use the img_id from ImageReference
            "image_content": image_ref.base64_data,
            "source_metadata": {
                "type": "image",
                "format": image_ref.img_format,
                "size_bytes": image_ref.size_bytes,
                "estimated_tokens": image_ref.estimated_tokens,
                "placeholder": image_ref.placeholder
            }
        }
        image_mappings.append(image_mapping)
    
    return chunk_mappings, image_mappings


def extract_images_from_html(html_content: str):
    """
    Extract base64 images from HTML and replace with placeholders.
    
    Args:
        html_content: HTML string potentially containing base64 images
        
    Returns:
        Tuple of:
            - cleaned_html: HTML with images replaced by placeholders
            - images: List of ImageReference objects
    """
    images = []
    
    soup = BeautifulSoup(html_content, features='html.parser')
    img_tags = soup.find_all('img')
    
    img_counter = 1
    
    for img in img_tags:
        src = img.get('src', '')
        if src.startswith('data:image/') and ';base64,' in src:
            try:
                # Extract format and base64 data
                format_start = src.find('data:image/') + len('data:image/')
                format_end = src.find(';', format_start)
                img_format = src[format_start:format_end].lower()
                base64_data = src[src.find(';base64,') + len(';base64,'):].strip()
                
                img_id = f"img_{img_counter:03d}"
                placeholder = f"[IMAGE_{img_id}]"
                
                # Validate base64 data
                decoded = base64.b64decode(base64_data, validate=True)
                size_bytes = len(decoded)
                
                estimated_tokens = estimate_image_tokens(size_bytes)
                
                image_ref = ImageReference(
                    img_id=img_id,
                    base64_data=base64_data,
                    format=img_format,
                    size_bytes=size_bytes,
                    estimated_tokens=estimated_tokens,
                    placeholder=placeholder
                )
                
                images.append(image_ref)
                img.replace_with(placeholder)
                img_counter += 1
                
            except Exception as e:
                logger.warning(f"Invalid base64 data for {img_id}: {e}")
                continue
    
    cleaned_html = str(soup)
    
    logger.info(f"Extracted {len(images)} images from HTML")
    for img in images:
        logger.debug(f"{img.img_id}: {img.format} format, {img.size_bytes} bytes, ~{img.estimated_tokens} tokens")
    
    return cleaned_html, images


def ensure_text_is_string(text) -> str:
    """
    Ensure text is a string, converting if necessary.
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', errors='ignore')
    else:
        return str(text)


def chunk_text_by_tokens(text: str, max_chunk_tokens: int, model_encoding: str = "cl100k_base") -> list:
    """
    Split text into chunks based on token count.
    
    Args:
        text: Input text to chunk
        max_chunk_tokens: Maximum tokens per chunk
        model_encoding: Tokenizer encoding to use
        
    Returns:
        List of text chunks
    """
    # Initialize tokenizer
    encoding = tiktoken.get_encoding(model_encoding)
    
    # Tokenize the entire text
    tokens = encoding.encode(text)
    
    # Split into chunks
    chunks = []
    for i in range(0, len(tokens), max_chunk_tokens):
        chunk_tokens = tokens[i:i + max_chunk_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks


def estimate_image_tokens(size_bytes: int) -> int:
    """
    Estimate token count for an image based on its size.
    
    Args:
        size_bytes: Size of the image in bytes
        
    Returns:
        Estimated token count
    """
    # This is a rough estimation based on common image token calculations
    # For GPT-4V, images typically use ~85 tokens per 512x512 tile
    # This is a simplified calculation
    kb_size = size_bytes / 1024
    
    if kb_size < 100:
        return 85  # Small image, likely one tile
    elif kb_size < 500:
        return 170  # Medium image, ~2 tiles
    elif kb_size < 1000:
        return 255  # Larger image, ~3 tiles
    else:
        return 425  # Very large image, ~5 tiles


class ImageReference:
    """
    Data class for storing image reference information.
    """
    def __init__(self, img_id, base64_data, format, size_bytes, estimated_tokens, placeholder):
        self.img_id = img_id
        self.base64_data = base64_data
        self.img_format = format
        self.size_bytes = size_bytes
        self.estimated_tokens = estimated_tokens
        self.placeholder = placeholder


def merge_results_with_source_tracking(results: list, strict: bool = True) -> tuple:
    """
    Merge extraction results with source tracking.
    
    Args:
        results: List of (chunk_mappings, image_mappings) tuples where both are dictionaries
        strict: Whether to use strict validation
        
    Returns:
        Tuple of (merged_chunks, merged_images) as dictionaries
    """
    merged_chunks = {}
    merged_images = {}
    
    for source_idx, (chunk_dict, image_dict) in enumerate(results):
        # Add source tracking to each chunk
        for chunk_id, chunk_data in chunk_dict.items():
            # Add source index to metadata
            chunk_data['metadata']['source_index'] = source_idx
            
            # Create a globally unique key if needed (to avoid collisions)
            global_chunk_id = f"source_{source_idx}_{chunk_id}"
            merged_chunks[global_chunk_id] = chunk_data
        
        # Add source tracking to each image
        for image_id, image_data in image_dict.items():
            # Add source index to metadata
            image_data['metadata']['source_index'] = source_idx
            
            # Create a globally unique key if needed (to avoid collisions)
            global_image_id = f"source_{source_idx}_{image_id}"
            merged_images[global_image_id] = image_data
    
    return merged_chunks, merged_images


# Helper function to access mappings by ID
def get_chunk_by_id(chunk_mappings: dict, chunk_id: str):
    """
    Retrieve a specific chunk by its ID.
    
    Args:
        chunk_mappings: Dictionary of chunk mappings
        chunk_id: The ID of the chunk to retrieve
        
    Returns:
        Chunk data or None if not found
    """
    return chunk_mappings.get(chunk_id)


def get_image_by_id(image_mappings: dict, image_id: str):
    """
    Retrieve a specific image by its ID.
    
    Args:
        image_mappings: Dictionary of image mappings
        image_id: The ID of the image to retrieve
        
    Returns:
        Image data or None if not found
    """
    return image_mappings.get(image_id)
