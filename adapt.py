#!/usr/bin/env python3
"""
Adapter to convert pandas DataFrame from hybrid search results 
into the format expected by the source-centric reconciliation pipeline.
"""
import pandas as pd
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def convert_dataframe_to_source_results(
    df: pd.DataFrame,
    companies: List[Dict[str, Any]],
    all_sources: Dict[str, str],
    relevance_score_column: str = "AverageVolume"  # or any column you want to use for relevance
) -> Dict[str, Any]:
    """
    Convert a pandas DataFrame from hybrid search into source-centric results for Stage 3.
    
    Args:
        df: DataFrame with columns like ISIN, MasterId, PrimaryMasterId, UBSId, BBTicker, etc.
        companies: List of companies from Stage 1 with source tracking
        all_sources: Dictionary of all source content (chunk_id/img_id -> content)
        relevance_score_column: Column to use for relevance scoring (optional)
        
    Returns:
        Source-centric results matching the expected format for Stage 3 reconciliation
    """
    source_results = {}
    
    # Group companies by their source_id
    companies_by_source = {}
    for company in companies:
        source_id = company.get("source_id", "")
        if source_id:
            if source_id not in companies_by_source:
                companies_by_source[source_id] = []
            companies_by_source[source_id].append(company)
    
    # Process each source
    for source_id, source_companies in companies_by_source.items():
        logger.info(f"Processing source {source_id} with {len(source_companies)} companies")
        
        # Get source content
        source_content = all_sources.get(source_id, "")
        source_type = "image" if source_id.startswith("img_") else "text"
        
        # Search for database candidates for each company in this source
        source_candidates = {}
        
        for company in source_companies:
            company_word = company.get("Word", "").strip()
            if not company_word:
                continue
            
            # Find matching rows in dataframe for this company
            candidates = _find_dataframe_matches(df, company, relevance_score_column)
            source_candidates[company_word] = candidates
        
        # Create source result
        source_results[source_id] = {
            "source_type": source_type,
            "content": source_content[:2000] if source_type == "text" else "[BASE64_IMAGE]",
            "full_content": source_content,
            "extracted_companies": source_companies,
            "database_candidates": source_candidates,
            "stats": {
                "companies_extracted": len(source_companies),
                "companies_with_candidates": sum(1 for c in source_candidates.values() if c),
                "total_candidates": sum(len(c) for c in source_candidates.values())
            }
        }
    
    return source_results


def _find_dataframe_matches(
    df: pd.DataFrame, 
    company: Dict[str, Any],
    relevance_score_column: str = None
) -> List[Dict[str, Any]]:
    """
    Find matching rows in the dataframe for a given company.
    
    Args:
        df: DataFrame with database results
        company: Company dict with extracted identifiers
        relevance_score_column: Column to use for relevance scoring
        
    Returns:
        List of candidate matches from the dataframe
    """
    candidates = []
    
    # Build search criteria from company identifiers
    search_criteria = []
    
    # Check each identifier type
    if company.get("ISIN_value"):
        isin_value = company["ISIN_value"].strip()
        if isin_value and "ISIN" in df.columns:
            search_criteria.append(("ISIN", isin_value))
    
    if company.get("Symbol_value"):
        symbol_value = company["Symbol_value"].strip()
        if symbol_value and "Symbol" in df.columns:
            search_criteria.append(("Symbol", symbol_value))
    
    if company.get("RIC_value"):
        ric_value = company["RIC_value"].strip()
        if ric_value and "RIC" in df.columns:
            search_criteria.append(("RIC", ric_value))
    
    if company.get("SEDOL_value"):
        sedol_value = company["SEDOL_value"].strip()
        if sedol_value and "SEDOL" in df.columns:
            search_criteria.append(("SEDOL", sedol_value))
    
    # Also search by company name in IssueName column
    company_word = company.get("Word", "").strip()
    if company_word and "IssueName" in df.columns:
        search_criteria.append(("IssueName", company_word))
    
    # Find matching rows for each criterion
    matched_indices = set()
    search_metadata = []
    
    for field, value in search_criteria:
        if field in df.columns:
            # Find exact matches
            mask = df[field].astype(str).str.strip() == value
            matches = df[mask]
            
            if not matches.empty:
                matched_indices.update(matches.index.tolist())
                search_metadata.append({
                    "matched_by_field": field,
                    "matched_by_value": value,
                    "match_count": len(matches)
                })
            
            # For company names, also try partial matching
            if field == "IssueName" and matches.empty:
                mask = df[field].astype(str).str.contains(value, case=False, na=False)
                matches = df[mask]
                if not matches.empty:
                    matched_indices.update(matches.index.tolist())
                    search_metadata.append({
                        "matched_by_field": f"{field}_partial",
                        "matched_by_value": value,
                        "match_count": len(matches)
                    })
    
    # Convert matched rows to candidate format
    for idx in matched_indices:
        row = df.loc[idx]
        
        # Calculate relevance score
        relevance_score = 1.0  # Default high score for any match
        if relevance_score_column and relevance_score_column in df.columns:
            try:
                # Normalize the column value to 0-1 range if possible
                col_value = float(row[relevance_score_column])
                max_value = df[relevance_score_column].max()
                if max_value > 0:
                    relevance_score = min(1.0, col_value / max_value)
            except (ValueError, TypeError):
                pass
        
        # Create candidate dict
        candidate = {
            "id": str(row.get("MasterId", row.get("PrimaryMasterId", f"idx_{idx}"))),
            "CompanyName": str(row.get("IssueName", company_word)),
            "RIC": str(row.get("RIC", "")) if pd.notna(row.get("RIC")) else "",
            "Symbol": str(row.get("Symbol", "")) if pd.notna(row.get("Symbol")) else "",
            "ISIN": str(row.get("ISIN", "")) if pd.notna(row.get("ISIN")) else "",
            "SEDOL": str(row.get("SEDOL", "")) if pd.notna(row.get("SEDOL")) else "",
            "BBTicker": str(row.get("BBTicker", "")) if pd.notna(row.get("BBTicker")) else "",
            "headquarters_country": str(row.get("CountryOfListing", "")) if pd.notna(row.get("CountryOfListing")) else "",
            "primary_exchange": _extract_exchange_from_ric(str(row.get("RIC", ""))),
            "relevance_score": relevance_score,
            "all_search_metadata": search_metadata.copy(),
            # Include additional fields from dataframe
            "MasterId": str(row.get("MasterId", "")) if pd.notna(row.get("MasterId")) else "",
            "PrimaryMasterId": str(row.get("PrimaryMasterId", "")) if pd.notna(row.get("PrimaryMasterId")) else "",
            "UBSId": str(row.get("UBSId", "")) if pd.notna(row.get("UBSId")) else "",
            "NonCompositeBBTicker": str(row.get("NonCompositeBBTicker", "")) if pd.notna(row.get("NonCompositeBBTicker")) else ""
        }
        
        candidates.append(candidate)
    
    # Sort by relevance score
    candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return candidates[:10]  # Return top 10 candidates


def _extract_exchange_from_ric(ric: str) -> str:
    """Extract exchange name from RIC code."""
    if not ric:
        return ""
    
    exchange_map = {
        '.O': 'NASDAQ',
        '.N': 'NYSE', 
        '.L': 'LSE',
        '.T': 'TSE',
        '.HK': 'HKSE',
        '.SS': 'SSE',
        '.SZ': 'SZSE',
        '.DE': 'XETRA',
        '.PA': 'Euronext Paris',
        '.MI': 'Borsa Italiana',
        '.AS': 'Euronext Amsterdam',
        '.SW': 'SIX Swiss'
    }
    
    for suffix, exchange in exchange_map.items():
        if ric.endswith(suffix):
            return exchange
    
    return ""


def integrate_dataframe_with_pipeline(
    df: pd.DataFrame,
    companies: List[Dict[str, Any]], 
    all_sources: Dict[str, str],
    reconciler,
    source_mapper,
    model: str = "gpt-4"
) -> Dict[str, Any]:
    """
    Integration function to use DataFrame results in the reconciliation pipeline.
    
    Args:
        df: DataFrame with hybrid search results
        companies: Companies from Stage 1
        all_sources: All source content
        reconciler: SourceAwareReconciliationEngine instance
        source_mapper: SourceMapper instance
        model: Model to use for reconciliation
        
    Returns:
        Stage 3 reconciliation results
    """
    import asyncio
    
    # Convert DataFrame to source-centric results
    source_results = convert_dataframe_to_source_results(df, companies, all_sources)
    
    # Use the reconciler's source-centric reconciliation
    async def run_reconciliation():
        return await reconciler.reconcile_by_source(
            source_results=source_results,
            all_sources=all_sources,
            source_mapper=source_mapper,
            model=model
        )
    
    # Run async reconciliation
    return asyncio.run(run_reconciliation())


# Example usage
def example_usage():
    """
    Example of how to use the dataframe adapter in your pipeline.
    """
    import pandas as pd
    from mixed_content_pipeline import MixedContentReconciliationPipeline
    from openai import AsyncOpenAI
    
    # Your DataFrame from hybrid search
    df = pd.read_csv("your_search_results.csv")  # or however you get your DataFrame
    
    # Initialize pipeline components
    llm = AsyncOpenAI(api_key="your-api-key")
    pipeline = MixedContentReconciliationPipeline(llm)
    
    # After Stage 1 extraction, instead of Stage 2 database search:
    # companies = stage1_result.get("companies", [])
    # all_sources = {...}  # Your source content
    
    # Use DataFrame adapter for Stage 2 results
    # source_results = convert_dataframe_to_source_results(df, companies, all_sources)
    
    # Then continue with Stage 3 reconciliation
    # stage3_result = await pipeline.reconciler.reconcile_by_source(
    #     source_results=source_results,
    #     all_sources=all_sources,
    #     source_mapper=pipeline.source_mapper,
    #     model="gpt-4"
    # )
    
    print("See code comments for integration details")


if __name__ == "__main__":
    example_usage()
