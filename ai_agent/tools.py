from ddgs import DDGS
from langchain.tools import tool
import requests


@tool
def search_tool(query: str) -> str:
    """Search the web for relevant information.

    Use this tool when you need up-to-date or externally sourced information.
    Provide a concise, specific search query.

    Args:
        query: The information to search for on the web.

    Returns:
        Up to 5 web results with their title, URL, and content snippet.
    """

    results = DDGS().text(
        query,
        max_results=5,
    )

    if not results:
        return "No results found."

    return "\n\n".join(
        f"Title: {result.get('title', '')}\n"
        f"URL: {result.get('href', '')}\n"
        f"Snippet: {result.get('body', '')}"
        for result in results
    )


@tool
def wiki_tool(query: str) -> str:
    """Search Wikipedia for background information about a topic.

    Use this tool when you need a concise, general-purpose overview of a
    topic from Wikipedia. The tool searches for the most relevant page and
    returns its introductory summary.

    Args:
        query: A specific topic or subject to look up on Wikipedia.

    Returns:
        The title and introductory summary of the most relevant Wikipedia
        page, limited to 1000 characters.
    """

    headers = {"User-Agent": "AI-Agent-Research/1.0 (contact@example.com)"}

    # 1. Find the page title corresponding to the search
    search_resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
        },
        headers=headers,
        timeout=10,
    )

    if search_resp.status_code != 200:
        return f"Wikipedia search failed with status {search_resp.status_code}"

    try:
        search_data = search_resp.json()
    except requests.exceptions.JSONDecodeError:
        return "Wikipedia returned an invalid response (not JSON)."

    results = search_data.get("query", {}).get("search", [])
    if not results:
        return "No Wikipedia results found."

    title = results[0]["title"]

    # 2. Retrieve the summary of the found page
    summary_resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
        },
        headers=headers,
        timeout=10,
    )

    try:
        summary_data = summary_resp.json()
    except requests.exceptions.JSONDecodeError:
        return "Wikipedia returned an invalid response (not JSON)."

    pages = summary_data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    extract = page.get("extract", "")

    if not extract:
        return f"No content found for page '{title}'."

    return f"Title: {title}\nSummary: {extract[:1000]}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather and today's forecast for a city.

    Use this tool for weather-related questions about a specific city.

    Args:
        city: The name of the city, such as "Paris", "London", or "Tokyo".

    Returns:
        The current weather conditions and today's forecast, including
        temperature, feels-like temperature, high and low temperatures,
        precipitation probability, and wind speed.
    """

    # 1. Find the city's coordinates
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_response = requests.get(
        geo_url,
        params={
            "name": city,
            "count": 1,
            "language": "fr",
            "format": "json",
        },
        timeout=10,
    )

    geo_response.raise_for_status()
    geo_data = geo_response.json()

    if not geo_data.get("results"):
        return f"Je n'ai pas trouvé la ville « {city} »."

    location = geo_data["results"][0]

    latitude = location["latitude"]
    longitude = location["longitude"]
    city_name = location["name"]
    country = location.get("country", "")

    # 2. Retrieve the weather
    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_response = requests.get(
        weather_url,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 1,
        },
        timeout=10,
    )

    weather_response.raise_for_status()
    weather = weather_response.json()

    current = weather["current"]
    daily = weather["daily"]

    weather_codes = {
        0: "ciel dégagé",
        1: "principalement dégagé",
        2: "partiellement nuageux",
        3: "couvert",
        45: "brouillard",
        48: "brouillard givrant",
        51: "bruine légère",
        53: "bruine modérée",
        55: "bruine forte",
        61: "pluie légère",
        63: "pluie modérée",
        65: "forte pluie",
        71: "neige légère",
        73: "neige modérée",
        75: "fortes chutes de neige",
        80: "averses légères",
        81: "averses modérées",
        82: "fortes averses",
        95: "orage",
        96: "orage avec grêle légère",
        99: "orage avec forte grêle",
    }

    description = weather_codes.get(
        current["weather_code"],
        "conditions météorologiques inconnues"
    )

    return (
        f"Météo aujourd'hui à {city_name}, {country} :\n"
        f"- Conditions : {description}\n"
        f"- Température : {current['temperature_2m']} °C\n"
        f"- Ressenti : {current['apparent_temperature']} °C\n"
        f"- Maximum : {daily['temperature_2m_max'][0]} °C\n"
        f"- Minimum : {daily['temperature_2m_min'][0]} °C\n"
        f"- Probabilité maximale de pluie : "
        f"{daily['precipitation_probability_max'][0]} %\n"
        f"- Vent : {current['wind_speed_10m']} km/h"
    )
