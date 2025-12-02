"""Credit and usage tracking functionality for Gradium API.

This module provides functions to retrieve information about API credit usage
and account quotas.

Functions:
    get: Retrieve current credit balance information.
    summary: Retrieve usage summary statistics.
"""

from . import client as gradium_client

ROUTE = "usages/"


async def get(client: "gradium_client.GradiumClient") -> dict:
    """Get the number of remaining and allocated credits.

    Retrieves the current credit balance information for the account,
    including how many credits are remaining and how many are allocated.

    Args:
        client: GradiumClient instance.

    Returns:
        Dictionary containing credit information with keys like:
        - remaining: Number of credits remaining
        - allocated: Number of credits allocated
        - Other account-specific credit information

    Raises:
        aiohttp.ClientError: If the API request fails.
    """
    return await client.get(f"{ROUTE}credits")


async def summary(client: "gradium_client.GradiumClient") -> dict:
    """Get usage summary statistics.

    Retrieves a summary of API usage across all features, including
    text-to-speech, speech recognition, and other services.

    Args:
        client: GradiumClient instance.

    Returns:
        Dictionary containing usage summary with statistics such as:
        - Total requests made
        - Usage by service/feature
        - Time period information
        - Other usage metrics

    Raises:
        aiohttp.ClientError: If the API request fails.
    """
    return await client.get(f"{ROUTE}summary")
